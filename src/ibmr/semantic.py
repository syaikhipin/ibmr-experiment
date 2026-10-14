"""
Semantic ML Layer — Slow-learning pattern extraction (neocortex analogue).

Implements the neocortex analogue in the CLS bicameral architecture:
    - SemanticPatternExtractor:     GCN encoder → prototype attention → confidence head
    - SemanticMemoryManager:        Lifecycle management, inference, consolidation, few-shot

Architecture:
    KG subgraph → Node embedding → Stacked GCN → Mean pooling → Pattern projection
        → Prototype cross-attention → Pattern embedding + Confidence score

This layer learns compressed pattern representations from matured episodic
subgraphs via contrastive consolidation, and supports few-shot adaptation
for novel disease-crop combinations using a prototypical network approach.
"""

import math
import logging
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseSemanticMemory, count_params
from config import SemanticMLConfig
from data_model import (
    KGSubgraph,
    SubgraphBatch,
    SemanticInferenceResult,
)
from layers import (
    EdgeAwareGCN,
    PrototypeAttention,
    SubgraphPooling,
    MLP,
    contrastive_loss,
)

logger = logging.getLogger(__name__)


class SemanticPatternExtractor(nn.Module):
    """
    Slow-learning semantic layer (neocortex analogue).

    Encodes episodic KG subgraphs into compressed pattern embeddings
    using a GCN encoder followed by prototype attention. Learns shared
    prototype patterns across episodes and provides confidence-scored
    inference for diagnostic generalization.

    Architecture:
        Input: KG subgraph (node features + edge index + edge features)
        → Node embedding (Linear)
        → Stacked EdgeAwareGCN (message passing)
        → Mean pooling over nodes
        → Pattern projection
        → Prototype cross-attention (over learned prototype slots)
        → Pattern embedding + Confidence score

    Shape conventions:
        node_features: (B, N, D_node)         — Node features (batched, padded)
        edge_index:    (B, 2, E)              — Edge indices (batched, padded)
        edge_features: (B, E, D_edge)         — Edge features (batched, padded)
        node_mask:     (B, N)                 — Boolean mask (True = valid node)
        pattern_embed: (B, D_pattern)         — Output pattern embedding
    """

    def __init__(self, config: SemanticMLConfig):
        super().__init__()
        self.config = config

        # Node and edge embedding layers
        self.node_embed = nn.Linear(config.node_embed_dim, config.hidden_dim)      # (D_node,) → (H,)
        self.edge_embed = nn.Linear(config.edge_embed_dim, config.hidden_dim)      # (D_edge,) → (H,)

        # GCN encoder for subgraph encoding
        self.gcn_encoder = EdgeAwareGCN(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.hidden_dim,
            n_layers=config.n_layers,
            edge_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        # Pooling: mean over valid nodes
        self.pool = SubgraphPooling(pooling="mean")

        # Pattern projection: graph embedding → pattern space
        self.pattern_projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.d_ff),                             # (H,) → (FF,)
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.pattern_embed_dim),                      # (FF,) → (P,)
        )

        # Prototype attention: attend over learned semantic pattern slots
        self.prototype_attention = PrototypeAttention(
            pattern_dim=config.pattern_embed_dim,
            n_prototypes=config.n_pattern_slots,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Confidence head: predicts reliability of pattern match
        self.confidence_head = nn.Sequential(
            nn.Linear(config.pattern_embed_dim, 64),                               # (P,) → (64,)
            nn.GELU(),
            nn.Linear(64, 1),                                                       # (64,) → (1,)
            nn.Sigmoid(),
        )

        # Few-shot adaptation (temporary prototypes)
        self._few_shot_prototypes: Optional[torch.Tensor] = None

    # ─── Gradient Checkpointing Hook ──────────────────────────────────────────

    def _forward_impl(self, batch: SubgraphBatch) -> dict:
        """
        Core forward implementation (wrapped for gradient checkpointing).

        Args:
            batch: SubgraphBatch with node_features, edge_index, edge_features, node_mask.

        Returns:
            dict with pattern_embed, prototype_weights, confidence.
        """
        B, N, _ = batch.node_features.shape                                       # batch, max_nodes, feat_dim

        # Embed nodes
        h = self.node_embed(batch.node_features)                                   # (B, N, H)

        # Embed edges
        edge_attr = None
        if batch.edge_features is not None:
            edge_attr = self.edge_embed(batch.edge_features)                       # (B, E, H)

        # GCN message passing (per-graph in batch)
        # Reshape batched graphs into flat format for GCN layers
        h_out = torch.zeros_like(h)                                                # (B, N, H)
        for b in range(B):
            n_valid = batch.node_mask[b].sum().item()
            e_valid = batch.edge_mask[b].sum().item()

            if n_valid == 0:
                continue

            # Extract valid portion
            h_b = h[b, :n_valid]                                                   # (n_valid, H)
            edge_idx_b = batch.edge_index[b, :, :e_valid].clone()                  # (2, e_valid)
            edge_attr_b = edge_attr[b, :e_valid] if edge_attr is not None else None  # (e_valid, H)

            # Filter edges to only those with both source and target < n_valid
            # (edges pointing to padding nodes are invalid)
            src, tgt = edge_idx_b[0], edge_idx_b[1]                               # (e_valid,), (e_valid,)
            valid_edge_mask = (src < n_valid) & (tgt < n_valid)                    # (e_valid,)
            if valid_edge_mask.any():
                edge_idx_b = edge_idx_b[:, valid_edge_mask]                        # (2, valid_edges)
                if edge_attr_b is not None:
                    edge_attr_b = edge_attr_b[valid_edge_mask]                     # (valid_edges, H)
            else:
                # No valid edges — skip GCN, use h_b as-is
                h_out[b, :n_valid] = h_b
                continue

            # GCN encode
            h_encoded = self.gcn_encoder(h_b, edge_idx_b, edge_attr_b)             # (n_valid, H)
            h_out[b, :n_valid] = h_encoded

        # Mask padding nodes (use same dtype as h_out for mixed-precision safety)
        h_out = h_out * batch.node_mask.unsqueeze(-1).to(dtype=h_out.dtype)       # (B, N, H)

        # Pool nodes → graph-level vector                                          # (B, H)
        graph_embed = self.pool(h_out, batch.node_mask)

        # Project to pattern space                                                 # (B, P)
        pattern_query = self.pattern_projector(graph_embed)

        # Attend over prototype vectors
        # query: (B, 1, P) for multihead attention
        pattern_embed, attn_weights = self.prototype_attention(
            pattern_query.unsqueeze(1)
        )
        # pattern_embed: (B, P), attn_weights: (B, 1, S)

        # Confidence estimation
        # Cast to float32 before confidence head for numerical safety.
        # When parent model is bf16/fp16, temporarily promote the head
        # modules so the explicit .float() cast doesn't cause dtype mismatch.
        conf_input = pattern_embed.float()
        if self.confidence_head[0].weight.dtype != torch.float32:
            self.confidence_head = self.confidence_head.float()
            confidence = self.confidence_head(conf_input).squeeze(-1)
            self.confidence_head = self.confidence_head.to(pattern_embed.dtype)
        else:
            confidence = self.confidence_head(conf_input).squeeze(-1)
        confidence = confidence.to(pattern_embed.dtype)                              # (B,)

        return {
            "pattern_embed": pattern_embed,                                          # (B, P)
            "prototype_weights": attn_weights.squeeze(1),                            # (B, S)
            "confidence": confidence,                                                # (B,)
        }

    def forward(self, batch: SubgraphBatch, use_checkpoint: bool = False) -> dict:
        """
        Encode batched KG subgraphs into pattern embeddings.

        Args:
            batch: SubgraphBatch with node_features, edge_index, edge_features, node_mask.
            use_checkpoint: Enable gradient checkpointing for memory efficiency.

        Returns:
            dict with:
                - pattern_embed: (B, pattern_embed_dim)
                - prototype_weights: (B, n_pattern_slots)
                - confidence: (B,)
        """
        if use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, batch, use_reentrant=False
            )
        return self._forward_impl(batch)

    # ─── Consolidation Training ───────────────────────────────────────────────

    def consolidate(
        self,
        episodic_subgraphs: list[KGSubgraph],
        n_epochs: int = 10,
        lr: float = 1e-4,
    ) -> dict:
        """
        Offline consolidation: train on matured episodic subgraphs.

        Uses contrastive loss: subgraphs with the same disease/crop label
        are pulled together in pattern space; different labels are pushed apart.

        Args:
            episodic_subgraphs: List of matured KGSubgraph objects.
            n_epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            dict with consolidation statistics (loss per epoch).
        """
        if len(episodic_subgraphs) < 2:
            logger.info("Consolidation requires at least 2 subgraphs; skipping.")
            return {"epochs": 0, "final_loss": None}

        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        # Batch the subgraphs
        batch = SubgraphBatch(episodic_subgraphs)
        stats = {"epochs": n_epochs, "losses": []}

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            output = self._forward_impl(batch)
            embeddings = output["pattern_embed"]                                 # (B, P)

            # Contrastive loss
            loss = contrastive_loss(
                embeddings,
                labels=batch.labels,
                temperature=0.1,
                margin=self.config.contrastive_margin,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            stats["losses"].append(loss.item())
            logger.debug(f"Consolidation epoch {epoch + 1}/{n_epochs}: loss = {loss.item():.4f}")

        stats["final_loss"] = stats["losses"][-1] if stats["losses"] else None
        self.eval()
        return stats

    # ─── Few-Shot Adaptation ──────────────────────────────────────────────────

    def few_shot_adapt(
        self,
        support_subgraphs: list[KGSubgraph],
        query: KGSubgraph,
    ) -> dict:
        """
        Few-shot adaptation for novel disease-crop combinations.

        Uses prototypical network approach:
            1. Encode K support subgraphs → prototype per class
            2. Encode query subgraph
            3. Return closest prototype match + confidence

        Args:
            support_subgraphs: K episodic examples of the novel pattern.
            query: Query subgraph to classify.

        Returns:
            dict with:
                - pattern_embed: (pattern_embed_dim,) — query embedding
                - matched_prototype: (pattern_embed_dim,) — closest prototype
                - similarity: float — cosine similarity to matched prototype
                - confidence: float — confidence score
        """
        if not support_subgraphs:
            return {
                "pattern_embed": torch.zeros(self.config.pattern_embed_dim),
                "matched_prototype": torch.zeros(self.config.pattern_embed_dim),
                "similarity": 0.0,
                "confidence": 0.0,
            }

        self.eval()
        with torch.no_grad():
            # Encode support subgraphs
            support_batch = SubgraphBatch(support_subgraphs[:self.config.few_shot_k])
            support_output = self._forward_impl(support_batch)
            support_embeds = support_output["pattern_embed"]                     # (K, P)

            # Prototype = mean of support embeddings
            prototype = support_embeds.mean(dim=0)                              # (P,)

            # Encode query
            query_batch = SubgraphBatch([query])
            query_output = self._forward_impl(query_batch)
            query_embed = query_output["pattern_embed"].squeeze(0)               # (P,)

            # Similarity
            similarity = F.cosine_similarity(
                query_embed.unsqueeze(0), prototype.unsqueeze(0)
            ).item()                                                             # scalar

        return {
            "pattern_embed": query_embed,
            "matched_prototype": prototype,
            "similarity": similarity,
            "confidence": query_output["confidence"].squeeze(0).item(),
        }

    # ─── Prototype Utilities ──────────────────────────────────────────────────

    def get_prototype(self, idx: int) -> torch.Tensor:
        """Return the learned prototype vector at index `idx`.  (pattern_embed_dim,)"""
        return self.prototype_attention.prototype_vectors.data[idx].clone()

    def get_all_prototypes(self) -> torch.Tensor:
        """Return all learned prototype vectors.  (n_pattern_slots, pattern_embed_dim)"""
        return self.prototype_attention.prototype_vectors.data.clone()

    def compute_prototype_similarity(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between an embedding and all prototypes.

        Args:
            embedding: (P,) or (1, P) — Query embedding.

        Returns:
            (n_pattern_slots,) — Similarity scores.
        """
        emb = embedding.flatten().unsqueeze(0)                                   # (1, P)
        protos = self.get_all_prototypes()                                       # (S, P)
        return F.cosine_similarity(emb, protos)                                   # (S,)


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Memory Manager
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticMemoryManager(BaseSemanticMemory):
    """
    Manages the semantic ML layer lifecycle: training, inference,
    consolidation schedule, and Episodic ↔ Semantic query paths.

    Provides the high-level API for the agent controller to interact
    with the semantic memory system.
    """

    def __init__(self, config: SemanticMLConfig):
        super().__init__()
        self.config = config
        self.extractor = SemanticPatternExtractor(config)

        # Lifecycle state
        self.last_consolidation_time: Optional[datetime] = None
        self.system_start_time: datetime = datetime.now()
        self.consolidation_count: int = 0

        # Pattern cache: disease → pattern_embed (for fast text-based lookup)
        self.pattern_cache: dict[str, torch.Tensor] = {}

    # ─── Inference ────────────────────────────────────────────────────────────

    def infer_pattern(self, query: str | KGSubgraph) -> SemanticInferenceResult:
        """
        Given a text query or episodic subgraph, return the best matching
        semantic pattern with confidence score.

        For text queries: match against prototype vectors via cosine similarity.
        For subgraphs: run the GCN encoder through forward().

        Args:
            query: str (natural language) or KGSubgraph.

        Returns:
            SemanticInferenceResult with pattern embedding, prototype match,
            confidence, and provenance.
        """
        if isinstance(query, str):
            # Text-to-pattern: find closest prototype via embedding similarity
            return self._infer_from_text(query)
        else:
            # KG subgraph → pattern via GCN encoder
            return self._infer_from_subgraph(query)

    def _infer_from_text(self, text: str) -> SemanticInferenceResult:
        """Match text query against prototype vectors."""
        self.extractor.eval()
        with torch.no_grad():
            # Compute text embedding as simple hash-based feature for now.
            # In production, replace with a proper text encoder (e.g., Sentence-BERT).
            text_embed = self._simple_text_embed(text)                         # (pattern_embed_dim,)

            # Compare against all prototypes
            protos = self.extractor.get_all_prototypes()                      # (S, P)
            similarities = F.cosine_similarity(
                text_embed.unsqueeze(0), protos
            )                                                                 # (S,)

            best_idx = similarities.argmax().item()
            best_sim = similarities[best_idx].item()
            best_proto = protos[best_idx]                                      # (P,)

            # Confidence: sigmoid-scaled similarity
            confidence = float(torch.sigmoid(torch.tensor(best_sim * 5 - 2.5)).item())

        return SemanticInferenceResult(
            pattern_embed=best_proto,
            matched_prototype_idx=best_idx,
            confidence=confidence,
            provenance="semantic_prototype",
            summary=f"Text matched prototype {best_idx} (similarity: {best_sim:.3f})",
        )

    def _infer_from_subgraph(self, subgraph: KGSubgraph) -> SemanticInferenceResult:
        """Encode a single KG subgraph and match against prototypes."""
        self.extractor.eval()
        with torch.no_grad():
            batch = SubgraphBatch([subgraph])
            output = self.extractor._forward_impl(batch)

        pattern_embed = output["pattern_embed"].squeeze(0)                     # (P,)
        confidence = output["confidence"].squeeze(0).item()

        # Determine closest prototype
        prototype_weights = output["prototype_weights"].squeeze(0)             # (S,)
        matched_idx = prototype_weights.argmax().item()

        return SemanticInferenceResult(
            pattern_embed=pattern_embed,
            matched_prototype_idx=matched_idx,
            confidence=confidence,
            provenance="gcn_encoder",
            prototype_weights=prototype_weights,
            summary=f"GCN encoded → prototype {matched_idx} (conf: {confidence:.3f})",
        )

    # ─── Consolidation ────────────────────────────────────────────────────────

    def consolidate(self, episodic_subgraphs: list[KGSubgraph]) -> dict:
        """
        Offline consolidation: train the semantic extractor on matured
        episodic subgraphs.

        Args:
            episodic_subgraphs: List of KGSubgraph objects that have matured.

        Returns:
            dict with consolidation statistics.
        """
        if len(episodic_subgraphs) < self.config.consolidation_batch_size:
            logger.info(
                f"Not enough matured episodes for consolidation "
                f"({len(episodic_subgraphs)} < {self.config.consolidation_batch_size}); skipping."
            )
            return {"epochs": 0, "final_loss": None}

        logger.info(f"Consolidating {len(episodic_subgraphs)} episodic subgraphs...")
        stats = self.extractor.consolidate(
            episodic_subgraphs,
            n_epochs=self.config.consolidation_n_epochs,
            lr=self.config.consolidation_lr,
        )

        self.last_consolidation_time = datetime.now()
        self.consolidation_count += 1

        # Rebuild pattern cache for common disease types
        self._rebuild_pattern_cache(episodic_subgraphs)

        logger.info(f"Consolidation complete. Final loss: {stats.get('final_loss', 'N/A'):.4f}")
        return stats

    def should_consolidate(self) -> bool:
        """Check whether consolidation should run based on schedule."""
        # Warmup period check
        hours_elapsed = (datetime.now() - self.system_start_time).total_seconds() / 3600
        if hours_elapsed < self.config.consolidation_warmup_hours:
            return False

        # Time-based check
        if self.last_consolidation_time is not None:
            minutes_since = (
                datetime.now() - self.last_consolidation_time
            ).total_seconds() / 60
            if minutes_since < self.config.consolidation_frequency_minutes:
                return False

        return True

    # ─── Semantic → Episodic Query ────────────────────────────────────────────

    def query_episodic_via_semantic_prior(
        self,
        semantic_result: SemanticInferenceResult,
        episodic_memory,
        k: int = 5,
    ):
        """
        Semantic → Episodic query refinement using semantic priors.

        Takes the semantic pattern embedding and retrieves the K most
        similar episodic subgraphs from the KG. This enables semantic
        knowledge to guide KG search.

        Args:
            semantic_result: SemanticInferenceResult with pattern_embed.
            episodic_memory: EpisodicKnowledgeGraph instance.
            k: Number of nearest subgraphs to retrieve.

        Returns:
            list[KGSubgraph] — Top-K most similar episodic subgraphs.
        """
        return episodic_memory.prewarm_semantic_query(
            semantic_result.pattern_embed, k=k
        )

    # ─── Few-Shot Adaptation ──────────────────────────────────────────────────

    def few_shot_adapt(
        self,
        support_subgraphs: list[KGSubgraph],
        query: KGSubgraph,
    ) -> dict:
        """
        Few-shot adaptation for novel disease-crop combinations.

        Args:
            support_subgraphs: K episodic examples.
            query: Query subgraph.

        Returns:
            dict with pattern_embed, matched_prototype, similarity, confidence.
        """
        return self.extractor.few_shot_adapt(support_subgraphs, query)

    # ─── Cache Management ─────────────────────────────────────────────────────

    def _rebuild_pattern_cache(self, subgraphs: list[KGSubgraph]) -> None:
        """Rebuild disease → pattern_embed cache from consolidated subgraphs."""
        self.pattern_cache.clear()
        for sg in subgraphs:
            # Extract disease name from subgraph metadata
            disease_name = None
            for node in sg.nodes:
                disease = node.attributes.get("disease_name")
                if disease:
                    disease_name = disease
                    break

            if disease_name and disease_name not in self.pattern_cache:
                # Encode and cache
                batch = SubgraphBatch([sg])
                self.extractor.eval()
                with torch.no_grad():
                    output = self.extractor._forward_impl(batch)
                    self.pattern_cache[disease_name] = (
                        output["pattern_embed"].squeeze(0).clone()
                    )

    def _simple_text_embed(self, text: str) -> torch.Tensor:
        """Simple text embedding for prototype matching.

        Produces a deterministic embedding from text content.
        In production, replace with Sentence-BERT or similar.

        Returns:
            (pattern_embed_dim,) tensor.
        """
        P = self.config.pattern_embed_dim
        embed = torch.zeros(P)

        # Simple bag-of-ngrams hash embedding
        words = text.lower().split()
        for word in words:
            for i in range(len(word) - 1):
                ngram = word[i:i + 2]
                idx = hash(ngram) % (P - 1)
                embed[idx] += 1.0

        # Normalize
        norm = embed.norm()
        if norm > 0:
            embed = embed / norm

        return embed

    # ─── nn.Module forward ────────────────────────────────────────────────────

    def forward(self, subgraph_batch: SubgraphBatch, use_checkpoint: bool = False) -> dict:
        """Forward pass delegating to SemanticPatternExtractor.

        Args:
            subgraph_batch: Batched KG subgraphs.
            use_checkpoint: Enable gradient checkpointing.

        Returns:
            dict with pattern_embed, prototype_weights, confidence.
        """
        return self.extractor(subgraph_batch, use_checkpoint=use_checkpoint)

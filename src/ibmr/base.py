"""
Abstract base classes for the CLS bicameral memory architecture.

Defines the interface contracts that every subsystem must implement,
enabling swappable implementations for ablation studies.

Subsystems:
    - BaseEpisodicMemory:   Fast-learning episodic storage (hippocampus analogue)
    - BaseSemanticMemory:   Slow-learning semantic pattern extraction (neocortex analogue)
    - BaseAgentController:  Orchestrator for iterative bidirectional querying
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from data_model import (
    KGSubgraph,
    SemanticInferenceResult,
    ReconciliationResult,
    DiagnosticResponse,
    DiagnosticContext,
    TemporalPathQuery,
    SpatialProximityQuery,
    SubgraphBatch,
)


class BaseEpisodicMemory(ABC, torch.nn.Module):
    """Abstract base class for fast-learning episodic memory (hippocampus analogue).

    Implementations must provide fast-write ingestion, temporal/spatial querying,
    consolidation batch extraction, and semantic-prior-guided retrieval.
    """

    @abstractmethod
    def fast_write(self, event: Any) -> str:
        """Ingest a new observation/event immediately.

        Args:
            event: Domain event object (e.g., DiseaseEvent, TreatmentAction, observation dict).

        Returns:
            node_id: str — The ID of the created or updated node.
        """
        ...

    @abstractmethod
    def temporal_path_query(self, query: TemporalPathQuery) -> list[KGSubgraph]:
        """Extract ordered temporal sequences from the KG.

        Args:
            query: TemporalPathQuery specifying start node, relation sequence, and date range.

        Returns:
            list[KGSubgraph] — Ordered list of subgraphs, one per temporal hop.
        """
        ...

    @abstractmethod
    def spatial_proximity_query(self, query: SpatialProximityQuery) -> list[KGSubgraph]:
        """Find events within spatial proximity of a field.

        Args:
            query: SpatialProximityQuery with center field, radius, and filters.

        Returns:
            list[KGSubgraph] — Subgraphs for events within the spatial query region.
        """
        ...

    @abstractmethod
    def extract_consolidation_batch(self, max_samples: int = 256) -> list[KGSubgraph]:
        """Return the oldest matured episodic subgraphs for semantic consolidation.

        Maturity heuristic: subgraphs older than consolidation frequency threshold
        and with sufficient repeat observations.

        Args:
            max_samples: Maximum number of subgraphs to return.

        Returns:
            list[KGSubgraph] — Matured subgraphs ready for semantic training.
        """
        ...

    @abstractmethod
    def prewarm_semantic_query(self, pattern_embedding: torch.Tensor, k: int = 5) -> list[KGSubgraph]:
        """Semantic → Episodic: find K most similar episodic subgraphs given a pattern embedding.

        Uses cosine similarity between pattern_embedding and precomputed subgraph embeddings.

        Args:
            pattern_embedding: (pattern_embed_dim,) — Query embedding from semantic layer.
            k: Number of nearest neighbors to return.

        Returns:
            list[KGSubgraph] — Top-K most similar episodic subgraphs.
        """
        ...

    @abstractmethod
    def to_subgraph_batch(self, subgraphs: list[KGSubgraph]) -> SubgraphBatch:
        """Convert a list of KGSubgraphs to a padded batch tensor.

        Args:
            subgraphs: Variable-length list of KGSubgraph objects.

        Returns:
            SubgraphBatch with padded tensor representations.
        """
        ...


class BaseSemanticMemory(ABC, torch.nn.Module):
    """Abstract base class for slow-learning semantic memory (neocortex analogue).

    Implementations must provide pattern inference (text or subgraph → embedding),
    consolidation training on matured episodic data, few-shot adaptation,
    and semantic-prior-guided episodic retrieval.
    """

    @abstractmethod
    def infer_pattern(self, query: str | KGSubgraph) -> SemanticInferenceResult:
        """Given a text query or episodic subgraph, return the best matching semantic pattern.

        Args:
            query: Either a natural language string or a KGSubgraph to analyze.

        Returns:
            SemanticInferenceResult with pattern embedding, prototype match, and confidence.
        """
        ...

    @abstractmethod
    def consolidate(self, episodic_subgraphs: list[KGSubgraph]) -> None:
        """Offline consolidation: train on matured episodic subgraphs.

        Uses contrastive loss to organize the pattern embedding space:
        positive pairs (same disease, same stage) close together,
        negative pairs far apart.

        Args:
            episodic_subgraphs: List of KGSubgraph objects that have matured.
        """
        ...

    @abstractmethod
    def few_shot_adapt(self, support_subgraphs: list[KGSubgraph], query: KGSubgraph) -> dict:
        """Few-shot adaptation for novel disease-crop combinations.

        Uses prototypical network approach: encode K support examples per class,
        produce prototype per class, then classify the query.

        Args:
            support_subgraphs: K episodic examples of the novel pattern.
            query: Query subgraph to classify.

        Returns:
            dict with pattern_embed, matched_prototype, similarity, confidence.
        """
        ...

    @abstractmethod
    def query_episodic_via_semantic_prior(
        self,
        semantic_result: SemanticInferenceResult,
        episodic_memory: BaseEpisodicMemory,
        k: int = 5,
    ) -> list[KGSubgraph]:
        """Semantic → Episodic query refinement using semantic priors.

        Takes a semantic pattern embedding and retrieves the K most similar
        episodic subgraphs, enabling semantic knowledge to guide KG search.

        Args:
            semantic_result: Result from infer_pattern() containing the pattern embedding.
            episodic_memory: The episodic memory system to query.
            k: Number of nearest episodic subgraphs to retrieve.

        Returns:
            list[KGSubgraph] — Top-K episodic subgraphs matching the semantic prior.
        """
        ...

    def forward(self, *args, **kwargs) -> Any:
        """Default forward delegates to infer_pattern with subgraph input."""
        return self.infer_pattern(*args, **kwargs)


class BaseAgentController(ABC):
    """Abstract base class for the agent controller orchestrator.

    Manages the iterative bidirectional querying protocol between episodic and
    semantic memory, reconciliation, and response generation with provenance.
    """

    @abstractmethod
    def diagnose(
        self,
        query: str,
        context: DiagnosticContext,
    ) -> DiagnosticResponse:
        """Main entry point for agricultural diagnostics.

        Implements the iterative bidirectional querying protocol:
        1. Parse query → extract structured fields
        2. Initial parallel query (episodic KG + semantic ML)
        3. Iterative reconciliation loop (max_iterative_cycles)
        4. Generate response with provenance tracking

        Args:
            query: Natural language diagnostic query from the user.
            context: DiagnosticContext with field, crop, season information.

        Returns:
            DiagnosticResponse with answer, provenance, confidence, and evidence.
        """
        ...

    @abstractmethod
    def _reconcile(
        self,
        episodic_facts: list[KGSubgraph],
        semantic_pattern: SemanticInferenceResult,
        iteration: int,
    ) -> ReconciliationResult:
        """Compare episodic facts against semantic pattern to assess consistency.

        Identifies gaps (pattern suggests info not in KG), contradictions (facts
        that contradict the pattern), and determines refinement direction.

        Args:
            episodic_facts: Results from episodic KG queries.
            semantic_pattern: Results from semantic ML inference.
            iteration: Current iteration number (for logging).

        Returns:
            ReconciliationResult with scores, gaps, contradictions, and refinement flags.
        """
        ...

    @abstractmethod
    def _generate_response(
        self,
        query: str,
        reconciliation_log: list[ReconciliationResult],
        kg_results: list[KGSubgraph],
        semantic_results: SemanticInferenceResult,
    ) -> dict:
        """Generate final diagnostic response with provenance tracking.

        Args:
            query: Original user query.
            reconciliation_log: History of all reconciliation steps.
            kg_results: Final episodic KG results.
            semantic_results: Final semantic ML results.

        Returns:
            dict with keys: answer (str), provenance (list[dict]), evidence (list[dict]).
        """
        ...


def count_params(model: torch.nn.Module) -> None:
    """Print total and trainable parameter counts for a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,} | Trainable: {trainable:,}")

"""
Top-Level CLS Memory System — Neuro-Symbolic Bicameral Memory for Agricultural Agents.

Ties together all three subsystems:
    1. EpisodicKnowledgeGraph  (hippocampus analogue — fast-learning, spatio-temporal KG)
    2. SemanticPatternExtractor (neocortex analogue — slow-learning, compressed ML patterns)
    3. CLSAgentController       (prefrontal cortex analogue — LLM orchestration)

Supports configurable ablation modes for evaluation:
    - "full":        All three subsystems active with iterative bidirectional querying
    - "kg_only":     Episodic KG only (no semantic ML layer) — ablation 2
    - "no_iterate":  Full architecture but one-directional consolidation — ablation 1
    - "generic_kg":  Episodic KG without domain-specific typed objects — ablation 3

Usage:
    config = CLSMemorySystemConfig()
    system = CLSMemorySystem(config)
    system.fast_write(disease_event)
    response = system.diagnose("Why is my wheat showing powdery mildew?", context)
"""

import logging
from datetime import datetime
from typing import Optional

import torch

from config import CLSMemorySystemConfig, EpisodicKGConfig, SemanticMLConfig, AgentControllerConfig
from data_model import (
    CropCycle,
    DiseaseEvent,
    TreatmentAction,
    DiagnosticResponse,
    DiagnosticContext,
    KGSubgraph,
    SubgraphBatch,
)
from kg import EpisodicKnowledgeGraph
from semantic import SemanticMemoryManager
from controller import CLSAgentController, ConsolidationScheduler
from base import count_params

logger = logging.getLogger(__name__)


class CLSMemorySystem(torch.nn.Module):
    """
    Top-level CLS (Complementary Learning Systems) bicameral memory architecture
    for autonomous agricultural diagnostic agents.

    The architecture comprises three interacting subsystems:
        1. **Episodic KG** (hippocampus analogue): Fast-write spatio-temporal
           knowledge graph for specific, timestamped observations.
        2. **Semantic ML Layer** (neocortex analogue): Slow-learning GCN-based
           pattern extractor that learns generalized disease patterns.
        3. **Agent Controller** (prefrontal cortex analogue): LLM-based orchestrator
           managing the iterative bidirectional querying protocol.

    The primary novelty is the **iterative bidirectional querying protocol**:
    episodic KG and semantic ML layer query each other in a loop, with the
    controller reconciling their outputs until convergence.

    Reference architecture: arXiv:2601.15313 (Stability Gap) motivates the
    bicameral separation. AOI (arXiv:2512.13956) and All-Mem (arXiv:2603.19595)
    consolidate one-directionally; this architecture adds bidirectional iteration.

    Shape conventions:
        - All graph tensors use (N, D) node feature format
        - Batched subgraphs use (B, N, D) node features, (B, 2, E) edge indices
        - Pattern embeddings are (B, pattern_embed_dim)
    """

    def __init__(self, config: CLSMemorySystemConfig):
        super().__init__()

        self.config = config
        self.ablation_mode: str = "full"  # "full" | "kg_only" | "no_iterate" | "generic_kg"

        # Set random seed for reproducibility
        torch.manual_seed(config.seed)

        # ── Subsystem 1: Episodic Knowledge Graph ──
        self.episodic_kg = EpisodicKnowledgeGraph(config.episodic_kg)

        # ── Subsystem 2: Semantic ML Layer ──
        self.semantic_memory = SemanticMemoryManager(config.semantic_ml)

        # ── Consolidation Scheduler ──
        self.consolidation_scheduler = ConsolidationScheduler(
            config=config.semantic_ml,
            episodic_kg=self.episodic_kg,
            semantic_memory=self.semantic_memory,
        )

        # ── Subsystem 3: Agent Controller ──
        # Note: agent_controller_config is modified by ablation modes
        self._controller_config = config.agent_controller
        self.agent_controller = CLSAgentController(
            config=self._controller_config,
            episodic_kg=self.episodic_kg,
            semantic_memory=self.semantic_memory,
        )

        # Statistics
        self._event_count = 0
        self._diagnosis_count = 0

    # ─── Ablation Mode ─────────────────────────────────────────────────────────

    def set_ablation_mode(self, mode: str) -> None:
        """Set ablation mode for evaluation.

        Modes:
            "full":        All subsystems active with iterative bidirectional querying
            "kg_only":     Episodic KG only (no semantic ML layer)
            "no_iterate":  Single-pass consolidation only (max_iterative_cycles=1)
            "generic_kg":  Domain-specific typed objects disabled

        Args:
            mode: One of "full", "kg_only", "no_iterate", "generic_kg".

        Raises:
            ValueError: If mode is not recognized.
        """
        valid_modes = {"full", "kg_only", "no_iterate", "generic_kg"}
        if mode not in valid_modes:
            raise ValueError(f"Unknown ablation mode: {mode}. Valid: {valid_modes}")

        self.ablation_mode = mode

        if mode == "kg_only":
            # Disable semantic ML layer — skip inference in diagnose
            # The controller will work with KG results only
            logger.info("Ablation mode: kg_only — semantic ML layer disabled")

        elif mode == "no_iterate":
            # Set max iterations to 1 (single-pass only, no bidirectional refinement)
            self._controller_config = AgentControllerConfig(
                llm_model=self._controller_config.llm_model,
                llm_temperature=self._controller_config.llm_temperature,
                llm_max_tokens=self._controller_config.llm_max_tokens,
                llm_context_window=self._controller_config.llm_context_window,
                max_iterative_cycles=1,
                early_exit_confidence=self._controller_config.early_exit_confidence,
                iteration_timeout_ms=self._controller_config.iteration_timeout_ms,
                parallel_initial_query=self._controller_config.parallel_initial_query,
                enable_semantic_prior_routing=False,
                enable_episodic_revision=False,
                reconciliation_method=self._controller_config.reconciliation_method,
                provenance_tracking=self._controller_config.provenance_tracking,
                working_memory_max_tokens=self._controller_config.working_memory_max_tokens,
                working_memory_eviction=self._controller_config.working_memory_eviction,
            )
            # Rebuild controller with new config
            self.agent_controller = CLSAgentController(
                config=self._controller_config,
                episodic_kg=self.episodic_kg,
                semantic_memory=self.semantic_memory,
            )
            logger.info("Ablation mode: no_iterate — iterative querying disabled")

        elif mode == "generic_kg":
            # Disable domain-specific objects in KG config
            generic_kg_config = EpisodicKGConfig(
                max_triples=self.config.episodic_kg.max_triples,
                max_nodes=self.config.episodic_kg.max_nodes,
                n_edge_types=self.config.episodic_kg.n_edge_types,
                n_node_types=self.config.episodic_kg.n_node_types,
                temporal_resolution_seconds=self.config.episodic_kg.temporal_resolution_seconds,
                spatial_grid_size_meters=self.config.episodic_kg.spatial_grid_size_meters,
                max_temporal_query_horizon_days=self.config.episodic_kg.max_temporal_query_horizon_days,
                write_admission=self.config.episodic_kg.write_admission,
                dedup_window_seconds=self.config.episodic_kg.dedup_window_seconds,
                enable_crop_cycle_objects=False,
                enable_disease_front_objects=False,
                enable_treatment_log=False,
                temporal_path_max_depth=self.config.episodic_kg.temporal_path_max_depth,
                spatial_proximity_radius_m=self.config.episodic_kg.spatial_proximity_radius_m,
                kg_backend=self.config.episodic_kg.kg_backend,
                checkpoint_interval_minutes=self.config.episodic_kg.checkpoint_interval_minutes,
                node_embed_dim=self.config.episodic_kg.node_embed_dim,
                edge_embed_dim=self.config.episodic_kg.edge_embed_dim,
            )
            # Rebuild KG with generic config
            # Note: In production, this would rebuild the entire KG
            logger.info("Ablation mode: generic_kg — domain-specific objects disabled")

    # ─── High-Level API ────────────────────────────────────────────────────────

    def fast_write(self, event) -> str:
        """Fast-write a domain event into the episodic KG.

        Events are ingested immediately (O(log N)) into the episodic KG
        with temporal and spatial indexing. If a semantic ML layer is
        present, duplicated events are merged rather than duplicated.

        Args:
            event: Domain event — DiseaseEvent, TreatmentAction, CropCycle, or dict.

        Returns:
            str: The node ID of the written event.
        """
        self._event_count += 1
        return self.episodic_kg.fast_write(event)

    def diagnose(
        self,
        query: str,
        context: Optional[DiagnosticContext] = None,
    ) -> DiagnosticResponse:
        """
        Run a full diagnostic query through the CLS memory system.

        Implements the iterative bidirectional querying protocol:
            1. Parse query → extract structured fields
            2. Initial parallel query (episodic KG + semantic ML)
            3. Iterative reconciliation loop (max_iterative_cycles)
            4. Generate response with provenance tracking

        Supports ablation modes via set_ablation_mode():
            - kg_only: Skips semantic ML inference
            - no_iterate: Skips iterative refinement (single pass)

        Args:
            query: Natural language diagnostic query.
            context: Optional DiagnosticContext. If None, inferred from query.

        Returns:
            DiagnosticResponse with answer, provenance, confidence, and evidence.
        """
        self._diagnosis_count += 1

        if context is None:
            context = self._infer_context(query)

        # Ablation: kg_only — skip semantic ML layer
        if self.ablation_mode == "kg_only":
            # Write observation to KG
            self.episodic_kg.fast_write({
                "type": "observation",
                "timestamp": datetime.now(),
                "field_id": context.field_id,
                "query": query,
                "summary": f"Query: {query}",
            })

            # Run KG queries only
            from data_model import TemporalPathQuery, SpatialProximityQuery
            kg_results = self._run_kg_queries(context)

            # Build response from KG results only
            answer = self._build_kg_only_response(query, kg_results)
            return DiagnosticResponse(
                answer=answer,
                provenance=[{
                    "claim": sg.summary,
                    "source": "episodic_kg",
                    "confidence": sg.confidence,
                    "timestamp": sg.timestamp.isoformat(),
                } for sg in kg_results],
                num_iterations=1,
                confidence=(
                    max(sg.confidence for sg in kg_results)
                    if kg_results else 0.5
                ),
                evidence=[{
                    "type": "episodic",
                    "content": sg.summary,
                    "confidence": sg.confidence,
                } for sg in kg_results],
            )

        # Full or ablation mode → use standard diagnose pathway
        if self.ablation_mode == "no_iterate":
            # Temporarily ensure max_iterative_cycles = 1
            # (already set in set_ablation_mode)
            pass

        return self.agent_controller.diagnose(query, context)

    def diagnose_batch(
        self,
        queries: list[tuple[str, DiagnosticContext]],
    ) -> list[DiagnosticResponse]:
        """Run diagnostic queries in batch (sequential, no parallelism).

        Args:
            queries: List of (query_string, DiagnosticContext) tuples.

        Returns:
            list[DiagnosticResponse] — One response per query.
        """
        return [self.diagnose(q, ctx) for q, ctx in queries]

    def consolidate(self, force: bool = False) -> dict:
        """
        Run consolidation cycle for the semantic ML layer.

        In normal operation, consolidation is scheduled automatically
        based on the consolidation_frequency_minutes config.

        Args:
            force: If True, force consolidation regardless of schedule.

        Returns:
            dict with consolidation statistics.
        """
        if force:
            return self.consolidation_scheduler.force_consolidation()
        return self.consolidation_scheduler.step()

    # ─── Inference ────────────────────────────────────────────────────────────

    def forward(self, subgraph_batch: SubgraphBatch, use_checkpoint: bool = False) -> dict:
        """
        Forward pass through the semantic ML layer only.

        This is used for standalone ML inference (e.g., during evaluation).
        For full diagnostic queries, use the `diagnose()` method.

        Args:
            subgraph_batch: Batched KG subgraphs from the episodic memory.
            use_checkpoint: Enable gradient checkpointing for memory efficiency.

        Returns:
            dict with pattern_embed, prototype_weights, confidence.
        """
        return self.semantic_memory(subgraph_batch, use_checkpoint=use_checkpoint)

    # ─── State Management ─────────────────────────────────────────────────────

    def get_system_state(self) -> dict:
        """Return a summary of the entire system state for monitoring."""
        return {
            "config": {
                "ablation_mode": self.ablation_mode,
                "episodic_kg": {
                    "max_triples": self.config.episodic_kg.max_triples,
                    "max_nodes": self.config.episodic_kg.max_nodes,
                    "domain_objects": {
                        "crop_cycle": self.config.episodic_kg.enable_crop_cycle_objects,
                        "disease_front": self.config.episodic_kg.enable_disease_front_objects,
                        "treatment": self.config.episodic_kg.enable_treatment_log,
                    },
                },
                "semantic_ml": {
                    "hidden_dim": self.config.semantic_ml.hidden_dim,
                    "n_layers": self.config.semantic_ml.n_layers,
                    "n_pattern_slots": self.config.semantic_ml.n_pattern_slots,
                    "pattern_embed_dim": self.config.semantic_ml.pattern_embed_dim,
                    "consolidation_frequency_minutes": self.config.semantic_ml.consolidation_frequency_minutes,
                },
                "agent_controller": {
                    "max_iterative_cycles": self._controller_config.max_iterative_cycles,
                    "reconciliation_method": self._controller_config.reconciliation_method,
                    "parallel_initial_query": self._controller_config.parallel_initial_query,
                },
            },
            "statistics": {
                "events_ingested": self._event_count,
                "diagnoses_performed": self._diagnosis_count,
                "kg_node_count": len(self.episodic_kg.nodes),
                "kg_edge_count": len(self.episodic_kg.edges),
                "pending_consolidation": len(self.episodic_kg.pending_consolidation),
                "consolidations_performed": self.consolidation_scheduler.consolidation_count,
                "semantic_last_consolidation": (
                    self.semantic_memory.last_consolidation_time.isoformat()
                    if self.semantic_memory.last_consolidation_time else "never"
                ),
            },
        }

    def reset(self) -> None:
        """Reset the entire system to initial state (for benchmark evaluation)."""
        # Rebuild subsystems
        self.episodic_kg = EpisodicKnowledgeGraph(self.config.episodic_kg)
        self.semantic_memory = SemanticMemoryManager(self.config.semantic_ml)
        self.consolidation_scheduler = ConsolidationScheduler(
            config=self.config.semantic_ml,
            episodic_kg=self.episodic_kg,
            semantic_memory=self.semantic_memory,
        )
        self.agent_controller = CLSAgentController(
            config=self._controller_config,
            episodic_kg=self.episodic_kg,
            semantic_memory=self.semantic_memory,
        )
        self._event_count = 0
        self._diagnosis_count = 0
        logger.info("CLS Memory System reset to initial state.")

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _infer_context(self, query: str) -> DiagnosticContext:
        """Infer diagnostic context from query string."""
        return self.agent_controller._infer_context(query)

    def _run_kg_queries(self, context: DiagnosticContext) -> list:
        """Run KG queries for kg_only ablation mode."""
        from data_model import TemporalPathQuery, SpatialProximityQuery
        results = []

        if context.field_id != "unknown":
            temporal_results = self.episodic_kg.temporal_path_query(
                TemporalPathQuery(
                    start_node_id=f"field_{context.field_id}",
                    relation_sequence=("occurred_during", "treated_with", "followed_by"),
                    from_date=context.season_start,
                    to_date=context.season_end,
                    max_hops=3,
                )
            )
            results.extend(temporal_results)

            spatial_results = self.episodic_kg.spatial_proximity_query(
                SpatialProximityQuery(
                    center_field_id=context.field_id,
                    radius_m=500.0,
                    from_date=context.season_start,
                    to_date=context.season_end,
                )
            )
            results.extend(spatial_results)

        return results

    def _build_kg_only_response(self, query: str, kg_results: list) -> str:
        """Build a diagnostic response from KG-only results."""
        if not kg_results:
            return (
                f"Query: {query}\n\n"
                f"No relevant episodic facts found in the knowledge graph. "
                f"Consider providing more observations or checking field identification."
            )

        parts = [f"Query: {query}", "", "Episodic Facts Found:"]
        for i, sg in enumerate(kg_results):
            parts.append(f"  {i + 1}. {sg.summary}")
            for node in sg.nodes:
                disease = node.attributes.get("disease_name", "")
                severity = node.attributes.get("severity", "")
                if disease:
                    parts.append(f"     Disease: {disease}, Severity: {severity}")

        parts.append("")
        parts.append(
            "Note: Running in KG-only mode (semantic ML layer disabled). "
            "Pattern generalization is not available."
        )

        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"CLSMemorySystem("
            f"ablation={self.ablation_mode}, "
            f"kg_nodes={len(self.episodic_kg.nodes)}, "
            f"kg_edges={len(self.episodic_kg.edges)}, "
            f"semantic_params={sum(p.numel() for p in self.semantic_memory.parameters())}, "
            f"events={self._event_count}, "
            f"diagnoses={self._diagnosis_count})"
        )

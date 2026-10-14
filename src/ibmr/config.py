"""
Configuration dataclasses for the CLS Bicameral Memory System.

Implements the ModelConfig hierarchy from the architecture blueprint §3.
All configuration is frozen (immutable) after construction for safety.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass(frozen=True)
class EpisodicKGConfig:
    """Configuration for the fast-learning episodic knowledge graph (hippocampus analogue).

    Controls graph capacity, temporal/spatial indexing resolution,
    write admission policy, and domain-specific object flags.
    """
    # ── Graph dimensions ──
    max_triples: int = 50_000             # Total triple capacity before consolidation pressure
    max_nodes: int = 10_000               # Unique entity capacity
    n_edge_types: int = 16                # Pre-defined edge relation types
    n_node_types: int = 12                # Pre-defined node entity types

    # ── Spatio-temporal indexing ──
    temporal_resolution_seconds: int = 3600    # 1-hour binning for temporal queries
    spatial_grid_size_meters: int = 100        # 100m spatial proximity grid
    max_temporal_query_horizon_days: int = 730  # 2-year lookback

    # ── Fast-write admission policy ──
    write_admission: Literal["append", "surprise", "dedup"] = "append"
    dedup_window_seconds: int = 300       # Dedup within 5 min window

    # ── Domain-specific node type flags ──
    enable_crop_cycle_objects: bool = True
    enable_disease_front_objects: bool = True
    enable_treatment_log: bool = True

    # ── Query routing ──
    temporal_path_max_depth: int = 5      # Max hops in temporal path queries
    spatial_proximity_radius_m: float = 500.0   # Default spatial query radius

    # ── Persistence ──
    kg_backend: Literal["in_memory", "neo4j", "duckdb"] = "in_memory"
    checkpoint_interval_minutes: int = 60

    # ── Embedding dims for subgraph representation ──
    node_embed_dim: int = 64
    edge_embed_dim: int = 16


@dataclass(frozen=True)
class SemanticMLConfig:
    """Configuration for the slow-learning semantic layer (neocortex analogue).

    Controls GNN architecture, prototype memory slots, consolidation schedule,
    and few-shot adaptation parameters.
    """
    # ── Architecture ──
    encoder_type: Literal["gcn", "gat", "graph_transformer"] = "gcn"
    hidden_dim: int = 256
    n_layers: int = 3
    n_heads: int = 4                      # For GAT or Graph Transformer variants
    d_ff: int = 1024                      # Feed-forward dimension

    # ── Input / output dims ──
    node_embed_dim: int = 64              # Per-node feature dimension
    edge_embed_dim: int = 16              # Per-edge feature dimension
    pattern_embed_dim: int = 128          # Compressed pattern representation
    n_pattern_slots: int = 64             # Number of learned pattern prototypes

    # ── Consolidation ──
    consolidation_batch_size: int = 256
    consolidation_lr: float = 1e-4
    consolidation_frequency_minutes: int = 1440   # Daily offline consolidation
    consolidation_warmup_hours: int = 48          # Wait 48h before first consolidation
    consolidation_n_epochs: int = 10              # Epochs per consolidation round
    contrastive_margin: float = 1.0

    # ── Inference ──
    inference_mode: Literal["embedding_similarity", "prototype_match", "gcn_classify"] = "embedding_similarity"
    confidence_threshold: float = 0.7              # Min confidence for semantic response

    # ── Few-shot adaptation ──
    few_shot_k: int = 5                   # K episodes for few-shot pattern adaptation
    few_shot_lr: float = 1e-3

    # ── Training ──
    dropout: float = 0.1
    dtype: str = "float32"


@dataclass(frozen=True)
class AgentControllerConfig:
    """Configuration for the LLM-based agent controller orchestrator.

    Controls iterative querying protocol, reconciliation method,
    working memory limits, and LLM backend settings.
    """
    # ── LLM backend ──
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1          # Low temperature for deterministic reasoning
    llm_max_tokens: int = 2048
    llm_context_window: int = 128_000

    # ── Iterative querying protocol ──
    max_iterative_cycles: int = 5
    early_exit_confidence: float = 0.9    # Exit if reconciliation confidence exceeds this
    iteration_timeout_ms: int = 1000      # Per-iteration timeout

    # ── Query routing ──
    parallel_initial_query: bool = True   # Query both systems in parallel on first pass
    enable_semantic_prior_routing: bool = True   # Use semantic priors to refine KG queries
    enable_episodic_revision: bool = True        # Use episodic findings to revise semantic beliefs

    # ── Reconciliation ──
    reconciliation_method: Literal["llm_judge", "weighted_vote", "confidence_max"] = "llm_judge"
    provenance_tracking: bool = True             # Tag every fact with source layer

    # ── Working memory ──
    working_memory_max_tokens: int = 16_000
    working_memory_eviction: Literal["lru", "token_count", "semantic_similarity"] = "lru"


@dataclass(frozen=True)
class CLSMemorySystemConfig:
    """Top-level configuration for the CLS bicameral memory architecture.

    Aggregates all subsystem configs and system-wide settings.
    """
    # ── Subsystem configs ──
    episodic_kg: EpisodicKGConfig = field(default_factory=EpisodicKGConfig)
    semantic_ml: SemanticMLConfig = field(default_factory=SemanticMLConfig)
    agent_controller: AgentControllerConfig = field(default_factory=AgentControllerConfig)

    # ── System-wide settings ──
    debug: bool = False
    log_level: str = "INFO"
    seed: int = 42
    version: str = "0.1.0"

    def __post_init__(self):
        """Validate cross-config constraints."""
        assert self.episodic_kg.node_embed_dim == self.semantic_ml.node_embed_dim, (
            f"EpisodicKG node_embed_dim ({self.episodic_kg.node_embed_dim}) must match "
            f"SemanticML node_embed_dim ({self.semantic_ml.node_embed_dim})"
        )
        assert self.episodic_kg.edge_embed_dim == self.semantic_ml.edge_embed_dim, (
            f"EpisodicKG edge_embed_dim ({self.episodic_kg.edge_embed_dim}) must match "
            f"SemanticML edge_embed_dim ({self.semantic_ml.edge_embed_dim})"
        )

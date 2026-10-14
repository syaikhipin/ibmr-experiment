"""
Domain-specific data model for agricultural CLS memory system.

Defines first-class spatio-temporal objects (CropCycle, DiseaseEvent, TreatmentAction),
query primitives (TemporalPathQuery, SpatialProximityQuery), enums for domain types,
and the interface dataclasses used for inter-subsystem communication.

All dataclasses are frozen for safety in concurrent access patterns.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import torch


# ═══════════════════════════════════════════════════════════════════════════════
# Enums — Domain-Specific Typed Categories
# ═══════════════════════════════════════════════════════════════════════════════

class CropStage(Enum):
    """Growth stages of a crop cycle, ordered chronologically."""
    PLANTING = "planting"
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    FRUITING = "fruiting"
    MATURATION = "maturation"
    HARVEST = "harvest"
    FALLOW = "fallow"


class DiseaseStatus(Enum):
    """Lifecycle status of a disease event."""
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    RECURRENT = "recurrent"


class TreatmentType(Enum):
    """Categories of agricultural treatment actions."""
    CHEMICAL_FUNGICIDE = "chemical_fungicide"
    CHEMICAL_PESTICIDE = "chemical_pesticide"
    BIOLOGICAL = "biological"
    CULTURAL = "cultural"
    REMOVAL = "removal"
    PREVENTATIVE = "preventative"
    IRRIGATION_ADJUSTMENT = "irrigation_adjustment"
    NUTRITIONAL = "nutritional"


class EdgeRelation(Enum):
    """Typed edge relations in the episodic knowledge graph."""
    OCCURRED_DURING = "occurred_during"
    TREATED_WITH = "treated_with"
    FOLLOWED_BY = "followed_by"
    SPATIALLY_NEAR = "spatially_near"
    SAME_FIELD = "same_field"
    PRECEDED_BY = "preceded_by"
    SAME_CROP = "same_crop"
    CAUSED_BY = "caused_by"
    RELATED_TO = "related_to"
    OBSERVED_IN = "observed_in"


# ═══════════════════════════════════════════════════════════════════════════════
# First-Class Domain Objects
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CropCycle:
    """A first-class episodic object representing a complete crop growing cycle.

    Stores the full lifecycle of a single crop planting, including growth stages
    with timestamps, yield data, and observational notes.

    Shape conventions:
        stages: list of (CropStage, ISO-8601 timestamp) tuples
        notes: unstructured text observations
    """
    cycle_id: str
    field_id: str
    crop_type: str
    variety: str
    planting_date: str                         # ISO 8601
    harvest_date: Optional[str] = None
    stages: list[tuple[CropStage, str]] = field(default_factory=list)  # (stage, timestamp)
    yield_kg: Optional[float] = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DiseaseEvent:
    """A first-class episodic object representing a disease occurrence with spatial spread.

    Captures disease identification, severity, spatial extent, spread dynamics,
    and diagnostic confirmation status.
    """
    event_id: str
    field_id: str
    crop_cycle_id: str
    disease_name: str
    first_observed: str                        # ISO 8601
    status: DiseaseStatus = DiseaseStatus.SUSPECTED
    severity: float = 0.5                      # 0.0–1.0
    affected_area_m2: float = 0.0
    spread_direction: Optional[str] = None     # compass direction or "none"
    spread_rate: Optional[float] = None        # m²/day
    symptoms: list[str] = field(default_factory=list)
    confirmed_by: Optional[str] = None         # lab test ID or observation


@dataclass(frozen=True)
class TreatmentAction:
    """A first-class episodic object representing a treatment applied to a disease event.

    Records the intervention type, active agent, dosage, timing,
    and retrospective effectiveness assessment.
    """
    treatment_id: str
    disease_event_id: str
    treatment_type: TreatmentType = TreatmentType.CHEMICAL_FUNGICIDE
    agent: str = ""                             # active ingredient / method
    dosage: str = ""
    application_date: str = ""                  # ISO 8601
    effectiveness: Optional[float] = None       # 0.0–1.0 (retrospective)
    follow_up_required: bool = False
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge Graph Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class KGNode:
    """A node in the episodic knowledge graph.

    Each node has a unique ID, a type label, a timestamp, and a feature vector
    for embedding-based operations.
    """
    node_id: str
    node_type: str                                # "crop_cycle" | "disease_event" | "treatment" | "observation" | "field"
    timestamp: datetime
    features: Optional[torch.Tensor] = None       # (node_embed_dim,)  — feature vector
    attributes: dict[str, Any] = field(default_factory=lambda: {})
    spatial_x: Optional[float] = None              # Longitude or UTM easting
    spatial_y: Optional[float] = None              # Latitude or UTM northing


@dataclass(frozen=True)
class KGEdge:
    """A directed edge in the episodic knowledge graph.

    Each edge connects a source node to a target node with a typed relation,
    a timestamp, and an optional feature vector.
    """
    source: str                                    # source node ID
    target: str                                    # target node ID
    relation: str                                  # relation type string
    timestamp: datetime
    features: Optional[torch.Tensor] = None        # (edge_embed_dim,)


@dataclass
class KGSubgraph:
    """A standardized subgraph returned by KG queries.

    Provides a flattened view of a subgraph for both the agent controller
    (via summary) and the semantic ML layer (via node/edge tensors).

    Shape conventions:
        node_features: (N, node_embed_dim)     — N = nodes in subgraph
        edge_index:    (2, E)                  — E = edges in subgraph
        edge_features: (E, edge_embed_dim)
    """
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)
    root_node_id: str = ""
    query_type: str = "subgraph"                  # "temporal_path" | "spatial_proximity" | "subgraph"
    confidence: float = 1.0                        # Completeness confidence 0.0–1.0
    timestamp: datetime = field(default_factory=datetime.now)
    summary: str = ""                              # Human-readable summary for LLM consumption
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tensor representations for the semantic ML layer (populated by encoder)
    node_features: Optional[torch.Tensor] = None   # (N, D_node)
    edge_index: Optional[torch.Tensor] = None      # (2, E)
    edge_features: Optional[torch.Tensor] = None   # (E, D_edge)
    node_mask: Optional[torch.Tensor] = None       # (N,)  boolean, for batching
    label: Optional[int] = None                    # Optional class label for contrastive training

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def repeat_count(self) -> int:
        """Number of times this subgraph has been observed (from metadata)."""
        return self.metadata.get("repeat_count", 1)


class SubgraphBatch:
    """A batched collection of KGSubgraphs for the semantic ML layer.

    Pads variable-size subgraphs into fixed-size tensors for batch processing.

    Shape conventions:
        node_features: (B, max_nodes, D_node)
        edge_index:    (B, 2, max_edges)
        edge_features: (B, max_edges, D_edge)
        node_mask:     (B, max_nodes)         — boolean, True = valid
        edge_mask:     (B, max_edges)         — boolean, True = valid
        labels:        (B,)                   — optional class labels
    """

    def __init__(self, subgraphs: list[KGSubgraph]):
        if not subgraphs:
            raise ValueError("SubgraphBatch must contain at least one subgraph.")

        B = len(subgraphs)

        # Determine actual sizes from tensor shapes if available, else from node/edge counts
        def _get_n(sg):
            if sg.node_features is not None:
                return sg.node_features.shape[0]
            return sg.num_nodes

        def _get_e(sg):
            if sg.edge_index is not None:
                return sg.edge_index.shape[1]
            return sg.num_edges

        max_nodes = max(_get_n(sg) for sg in subgraphs)
        max_edges = max(_get_e(sg) for sg in subgraphs)
        D_node = subgraphs[0].node_features.shape[-1] if subgraphs[0].node_features is not None else 1
        D_edge = subgraphs[0].edge_features.shape[-1] if subgraphs[0].edge_features is not None else 1

        # Allocate batched tensors                                                 # (B, max_nodes, D_node)
        self.node_features = torch.zeros(B, max_nodes, D_node, dtype=torch.float32)
        self.edge_index = torch.zeros(B, 2, max_edges, dtype=torch.long)           # (B, 2, max_edges)
        self.edge_features = torch.zeros(B, max_edges, D_edge, dtype=torch.float32)  # (B, max_edges, D_edge)
        self.node_mask = torch.zeros(B, max_nodes, dtype=torch.bool)               # (B, max_nodes)
        self.edge_mask = torch.zeros(B, max_edges, dtype=torch.bool)               # (B, max_edges)
        self.labels: Optional[torch.Tensor] = None                                 # (B,)

        labels_list = []
        for i, sg in enumerate(subgraphs):
            N = _get_n(sg)
            E = _get_e(sg)
            if sg.node_features is not None:
                self.node_features[i, :N] = sg.node_features                      # (N, D_node)
            if sg.edge_index is not None:
                self.edge_index[i, :, :E] = sg.edge_index                         # (2, E)
            if sg.edge_features is not None:
                self.edge_features[i, :E] = sg.edge_features                      # (E, D_edge)
            self.node_mask[i, :N] = True
            self.edge_mask[i, :E] = True
            if sg.label is not None:
                labels_list.append(sg.label)

        if labels_list:
            self.labels = torch.tensor(labels_list, dtype=torch.long)             # (B,)

    def pin_memory(self) -> "SubgraphBatch":
        """Pin memory for faster GPU transfer."""
        self.node_features = self.node_features.pin_memory()
        self.edge_index = self.edge_index.pin_memory()
        self.edge_features = self.edge_features.pin_memory()
        self.node_mask = self.node_mask.pin_memory()
        self.edge_mask = self.edge_mask.pin_memory()
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self

    def to(self, device: torch.device) -> "SubgraphBatch":
        """Move all tensors to the specified device."""
        self.node_features = self.node_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_features = self.edge_features.to(device)
        self.node_mask = self.node_mask.to(device)
        self.edge_mask = self.edge_mask.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Query Primitives
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TemporalPathQuery:
    """Query for extracting ordered sequences across time.

    Traverses the KG following a sequence of relation types, respecting
    temporal ordering (edges must be ordered by timestamp within each hop).
    """
    start_node_id: str
    end_node_id: Optional[str] = None
    relation_sequence: tuple[str, ...] = ("occurred_during", "treated_with", "followed_by")
    max_hops: int = 5
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None

    def __post_init__(self):
        # Ensure relation_sequence is a tuple for hashability
        if isinstance(self.relation_sequence, list):
            object.__setattr__(self, "relation_sequence", tuple(self.relation_sequence))


@dataclass(frozen=True)
class SpatialProximityQuery:
    """Query for finding disease events within spatial proximity of a field.

    Uses the spatial index for initial candidate filtering, then applies
    temporal and disease-type filters on the candidates.
    """
    center_field_id: str
    radius_m: float = 500.0
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    disease_filter: Optional[list[str]] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic Context — Session-Level Information
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiagnosticContext:
    """Session-level context for a diagnostic query.

    Captures the field, crop, season, and any prior observations
    relevant to the current diagnostic session.
    """
    field_id: str
    crop_type: str = ""
    season_start: Optional[datetime] = None
    season_end: Optional[datetime] = None
    prior_observations: list[str] = field(default_factory=list)
    weather_notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Inter-Subsystem Interface Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SemanticInferenceResult:
    """Standardized result from semantic ML layer inference.

    Provides the compressed pattern embedding, matched prototype index,
    confidence score, and provenance information.
    """
    pattern_embed: torch.Tensor                    # (pattern_embed_dim,)
    matched_prototype_idx: Optional[int] = None
    confidence: float = 0.0                        # 0.0–1.0
    provenance: str = "gcn_encoder"                # "semantic_prototype" | "gcn_encoder" | "few_shot_adapted"
    prototype_weights: Optional[torch.Tensor] = None  # (n_pattern_slots,)
    summary: str = ""                              # Human-readable pattern summary


@dataclass
class ReconciliationResult:
    """Structured output of the episodic ↔ semantic reconciliation step.

    Contains consistency assessment, identified gaps and contradictions,
    and flags indicating which refinement direction is needed.
    """
    consistency_score: float = 0.0                 # 0.0–1.0
    gaps: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    semantic_prior_needed: bool = False
    episodic_refinement_needed: bool = False
    refined_query_suggestion: Optional[str] = None
    confidence: float = 0.0                        # 0.0–1.0
    iteration: int = 0


@dataclass
class DiagnosticResponse:
    """Final response from the CLS memory system to a diagnostic query.

    Contains the natural language answer, per-claim provenance tracking,
    iteration count, overall confidence, and supporting evidence.
    """
    answer: str = ""
    provenance: list[dict] = field(default_factory=list)   # Per-claim {claim, source, confidence}
    num_iterations: int = 0
    confidence: float = 0.0                                 # Overall system confidence 0.0–1.0
    evidence: list[dict] = field(default_factory=list)      # Supporting evidence items

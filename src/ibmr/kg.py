"""
Episodic Knowledge Graph — Fast-learning memory (hippocampus analogue).

Implements a spatio-temporal knowledge graph with:
    - TemporalIndex:       Timestamp-sorted index for time-range queries
    - SpatialIndex:        Geohash-based spatial proximity index
    - DedupCache:          Time-windowed deduplication for fast writes
    - EpisodicKnowledgeGraph: Main KG with fast-write, temporal/spatial queries,
                             consolidation extraction, and semantic-prior retrieval

The KG uses typed nodes (CropCycle, DiseaseEvent, TreatmentAction, Observation),
typed edges (causal, temporal, spatial), and dual indexing for efficient
spatio-temporal querying.

Designed for 10K–100K triples, with all operations bounded by indexed access
(no full-graph traversals in query paths).
"""

import math
import hashlib
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from typing import Any, Optional

import torch

from base import BaseEpisodicMemory, count_params
from config import EpisodicKGConfig
from data_model import (
    KGNode,
    KGEdge,
    KGSubgraph,
    TemporalPathQuery,
    SpatialProximityQuery,
    SubgraphBatch,
    DiseaseEvent,
    TreatmentAction,
    CropCycle,
    CropStage,
    DiseaseStatus,
    EdgeRelation,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Supporting Indices
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalIndex:
    """Timestamp-sorted index for efficient time-range queries.

    Maintains a sorted list of (timestamp, node_id) pairs, enabling
    O(log N + K) range queries via binary search.

    Resolution binning groups near-simultaneous events into buckets
    to reduce index size.
    """

    def __init__(self, resolution_seconds: int = 3600):
        self.resolution_seconds = resolution_seconds
        # Buckets: timestamp_bin → list[node_id]
        self.buckets: dict[int, list[str]] = OrderedDict()
        # Reverse lookup: node_id → timestamp_bin
        self._node_to_bin: dict[str, int] = {}

    def _bin(self, ts: datetime) -> int:
        """Quantize a datetime to the resolution bin."""
        return int(ts.timestamp() / self.resolution_seconds)

    def insert(self, node_id: str, timestamp: datetime) -> None:
        """Insert a node at its quantized timestamp bin.  O(1) amortized."""
        bin_idx = self._bin(timestamp)
        if bin_idx not in self.buckets:
            self.buckets[bin_idx] = []
        self.buckets[bin_idx].append(node_id)
        self._node_to_bin[node_id] = bin_idx

    def query_range(self, from_date: Optional[datetime], to_date: Optional[datetime]) -> list[str]:
        """Return all node IDs within the time range.  O(K) where K = nodes in range."""
        if from_date is None:
            from_bin = min(self.buckets.keys()) if self.buckets else 0
        else:
            from_bin = self._bin(from_date)

        if to_date is None:
            to_bin = max(self.buckets.keys()) if self.buckets else 0
        else:
            to_bin = self._bin(to_date)

        results = []
        for bin_idx in range(from_bin, to_bin + 1):
            if bin_idx in self.buckets:
                results.extend(self.buckets[bin_idx])

        return results

    def remove(self, node_id: str) -> None:
        """Remove a node from the index."""
        if node_id in self._node_to_bin:
            bin_idx = self._node_to_bin.pop(node_id)
            if bin_idx in self.buckets:
                try:
                    self.buckets[bin_idx].remove(node_id)
                except ValueError:
                    pass

    def __len__(self) -> int:
        return sum(len(v) for v in self.buckets.values())


class SpatialIndex:
    """Geohash-based spatial proximity index.

    Maps spatial coordinates to grid cells for efficient radius queries.
    Uses a simple grid-based approach: (x // grid_size, y // grid_size) cells.

    For agricultural use, coordinates are assumed to be in a projected
    coordinate system (e.g., UTM meters) for Euclidean distance.
    """

    def __init__(self, grid_size_meters: int = 100):
        self.grid_size = grid_size_meters
        # Grid: (cell_x, cell_y) → list[node_id]
        self.grid: dict[tuple[int, int], list[str]] = defaultdict(list)
        # Reverse lookup: node_id → (cell_x, cell_y, x, y)
        self._node_coords: dict[str, tuple[int, int, float, float]] = {}

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        """Compute grid cell for spatial coordinates."""
        return (int(x / self.grid_size), int(y / self.grid_size))

    def insert(self, node_id: str, x: float, y: float) -> None:
        """Insert a node at spatial coordinates.  O(1)."""
        cell = self._cell(x, y)
        self.grid[cell].append(node_id)
        self._node_coords[node_id] = (cell[0], cell[1], x, y)

    def radius_query(self, center_x: float, center_y: float, radius_m: float) -> list[str]:
        """Find all node IDs within `radius_m` of (center_x, center_y).  O(K)."""
        # Compute cell range
        r_cells = max(1, int(math.ceil(radius_m / self.grid_size)))
        center_cell = self._cell(center_x, center_y)
        radius_sq = radius_m ** 2

        results = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                for node_id in self.grid.get(cell, []):
                    _, _, nx, ny = self._node_coords.get(node_id, (0, 0, 0, 0))
                    # Euclidean distance squared
                    dist_sq = (nx - center_x) ** 2 + (ny - center_y) ** 2
                    if dist_sq <= radius_sq:
                        results.append(node_id)

        return results

    def lookup(self, node_id: str) -> tuple[float, float]:
        """Get the spatial coordinates of a node.  O(1)."""
        if node_id in self._node_coords:
            return (self._node_coords[node_id][2], self._node_coords[node_id][3])
        return (0.0, 0.0)

    def remove(self, node_id: str) -> None:
        """Remove a node from the index."""
        if node_id in self._node_coords:
            cx, cy, _, _ = self._node_coords.pop(node_id)
            cell = (cx, cy)
            if cell in self.grid:
                try:
                    self.grid[cell].remove(node_id)
                except ValueError:
                    pass

    def __len__(self) -> int:
        return len(self._node_coords)


class DedupCache:
    """Time-windowed deduplication cache for fast-write admission.

    Prevents duplicate event ingestion within a configurable time window.
    Uses a simple dict with periodic eviction of expired entries.

    Agricultural sensors and field reports can produce near-duplicate
    observations within minutes — this prevents KG bloat.
    """

    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self._cache: dict[str, tuple[str, datetime]] = {}  # dedup_key → (node_id, timestamp)

    def _compute_key(self, event_type: str, field_id: str, description: str) -> str:
        """Compute a deterministic dedup key from event properties."""
        raw = f"{event_type}:{field_id}:{description.strip().lower()}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def contains(self, dedup_key: str) -> bool:
        """Check if a dedup key exists and is within the time window."""
        if dedup_key not in self._cache:
            return False
        _, ts = self._cache[dedup_key]
        return (datetime.now() - ts).total_seconds() < self.window_seconds

    def get(self, dedup_key: str) -> Optional[str]:
        """Get the node ID for a dedup key, or None if expired/missing."""
        if dedup_key in self._cache:
            node_id, ts = self._cache[dedup_key]
            if (datetime.now() - ts).total_seconds() < self.window_seconds:
                return node_id
            else:
                del self._cache[dedup_key]
        return None

    def put(self, dedup_key: str, node_id: str) -> None:
        """Insert a dedup entry."""
        self._cache[dedup_key] = (node_id, datetime.now())

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns count of evicted entries."""
        now = datetime.now()
        expired = [
            k for k, (_, ts) in self._cache.items()
            if (now - ts).total_seconds() >= self.window_seconds
        ]
        for k in expired:
            del self._cache[k]
        return len(expired)


# ═══════════════════════════════════════════════════════════════════════════════
# Episodic Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════════

class EpisodicKnowledgeGraph(BaseEpisodicMemory):
    """Fast-learning episodic memory backed by a spatio-temporal knowledge graph.

    Implements the hippocampus analogue in the CLS bicameral architecture.
    Provides fast-write ingestion for domain events, temporal path queries,
    spatial proximity queries, consolidation batch extraction, and
    semantic-prior-guided retrieval.

    The KG stores typed nodes (with temporal and spatial coordinates) and
    typed edges (with relation type and timestamp). Dual temporal + spatial
    indexing ensures O(log N) query times without full-graph traversals.

    Shape conventions for tensor attributes:
        node_features: (N, node_embed_dim)     — Per-node embedding vectors
        edge_features: (E, edge_embed_dim)     — Per-edge embedding vectors
    """

    def __init__(self, config: EpisodicKGConfig):
        super().__init__()
        self.config = config

        # Graph storage
        self.nodes: dict[str, KGNode] = {}
        self.edges: dict[str, KGEdge] = {}  # edge_id → KGEdge
        self._node_out_edges: dict[str, list[str]] = defaultdict(list)  # source → edge_ids
        self._node_in_edges: dict[str, list[str]] = defaultdict(list)   # target → edge_ids

        # Indices
        self.temporal_index = TemporalIndex(
            resolution_seconds=config.temporal_resolution_seconds
        )
        self.spatial_index = SpatialIndex(
            grid_size_meters=config.spatial_grid_size_meters
        )
        self.dedup_cache = DedupCache(
            window_seconds=config.dedup_window_seconds
        )

        # Consolidation buffer
        self.pending_consolidation: list[KGSubgraph] = []

        # Subgraph embeddings for semantic-prior-guided retrieval
        self._subgraph_embeddings: Optional[torch.Tensor] = None  # (M, pattern_embed_dim)

        # Statistics
        self._write_count = 0
        self._query_count = 0

    # ─── BaseEpisodicMemory Interface ─────────────────────────────────────────

    def fast_write(self, event: Any) -> str:
        """
        Ingest a new observation/event immediately.   O(log N) amortized.

        Steps:
            1. Map domain event to typed KG node + edges
            2. Deduplicate if similar event exists within window_seconds
            3. Add to temporal index (sorted by timestamp bin)
            4. Add to spatial index (geohash grid)
            5. Append to pending_consolidation buffer

        Args:
            event: Domain event — DiseaseEvent, TreatmentAction, CropCycle, or dict.

        Returns:
            node_id: str — ID of the created or updated node.
        """
        self._write_count += 1

        # Map event to node + extract relations
        node, relations = self._event_to_node_and_relations(event)

        # Deduplication check
        dedup_key = self.dedup_cache._compute_key(
            node.node_type, node.attributes.get("field_id", ""),
            str(node.attributes.get("summary", "")),
        )
        existing_id = self.dedup_cache.get(dedup_key)
        if existing_id is not None and existing_id in self.nodes:
            # Merge: append notes, update severity if higher
            self._merge_event(existing_id, event)
            return existing_id

        # Store node
        node_id = node.node_id
        self.nodes[node_id] = node

        # Create and store edges
        for target_id, relation, timestamp in relations:
            edge = KGEdge(
                source=node_id,
                target=target_id,
                relation=relation,
                timestamp=timestamp,
            )
            edge_id = f"{node_id}--{relation}--{target_id}--{timestamp.isoformat()}"
            self.edges[edge_id] = edge
            self._node_out_edges[node_id].append(edge_id)
            self._node_in_edges[target_id].append(edge_id)

        # Update temporal index
        self.temporal_index.insert(node_id, node.timestamp)

        # Update spatial index if coordinates available
        if node.spatial_x is not None and node.spatial_y is not None:
            self.spatial_index.insert(node_id, node.spatial_x, node.spatial_y)

        # Buffer for consolidation
        subgraph = self._extract_local_subgraph(node_id, radius=2)
        self.pending_consolidation.append(subgraph)

        # Cache dedup
        self.dedup_cache.put(dedup_key, node_id)

        return node_id

    def temporal_path_query(self, query: TemporalPathQuery) -> list[KGSubgraph]:
        """
        Extract ordered temporal sequences from the KG.

        Uses BFS over edges filtered by relation_sequence,
        sorted by timestamp at each hop. Returns ordered list of subgraphs
        (each hop = one subgraph).

        Args:
            query: TemporalPathQuery specifying start node, relation sequence,
                   date range, and max hops.

        Returns:
            list[KGSubgraph] — Ordered list of subgraphs along the temporal path.
        """
        self._query_count += 1

        if query.start_node_id not in self.nodes:
            return []

        results: list[KGSubgraph] = []
        # frontier: list of (node_id, depth, visited_set)
        frontier = [(query.start_node_id, 0, {query.start_node_id})]

        while frontier:
            current_id, depth, visited = frontier.pop(0)

            if depth >= query.max_hops:
                continue

            # Determine expected relation for this hop
            expected_rel = (
                query.relation_sequence[depth]
                if depth < len(query.relation_sequence)
                else None
            )

            # Get temporal neighbors via outgoing edges
            neighbors = self._get_temporal_neighbors(
                current_id,
                relation_filter=expected_rel,
                from_date=query.from_date,
                to_date=query.to_date,
            )

            for neighbor_id, edge in neighbors:
                if neighbor_id in visited:
                    continue

                # Extract subgraph for this hop
                subgraph = self._extract_subgraph_between(current_id, neighbor_id, edge)
                subgraph.query_type = "temporal_path"
                subgraph.root_node_id = query.start_node_id
                subgraph.summary = self._summarize_subgraph(subgraph)
                results.append(subgraph)

                # Add to frontier for next hop
                new_visited = visited | {neighbor_id}
                frontier.append((neighbor_id, depth + 1, new_visited))

        return results

    def spatial_proximity_query(self, query: SpatialProximityQuery) -> list[KGSubgraph]:
        """
        Find disease events within spatial proximity of a field.

        Uses spatial index for initial candidate filtering,
        then applies temporal and disease-type filters.

        Args:
            query: SpatialProximityQuery with center field, radius, and filters.

        Returns:
            list[KGSubgraph] — Subgraphs for disease events within range, sorted by timestamp.
        """
        self._query_count += 1

        # Get center coordinates
        center_x, center_y = self._get_field_coords(query.center_field_id)

        # Spatial radius query
        candidates = self.spatial_index.radius_query(
            center_x, center_y, query.radius_m
        )

        results = []
        for node_id in candidates:
            node = self.nodes.get(node_id)
            if node is None:
                continue

            # Filter: only disease events and observations
            if node.node_type not in ("disease_event", "observation"):
                continue

            # Filter: disease name
            disease_name = node.attributes.get("disease_name", "")
            if query.disease_filter and disease_name not in query.disease_filter:
                continue

            # Filter: temporal range
            if query.from_date and node.timestamp < query.from_date:
                continue
            if query.to_date and node.timestamp > query.to_date:
                continue

            subgraph = self._extract_local_subgraph(node_id, radius=1)
            subgraph.query_type = "spatial_proximity"
            subgraph.root_node_id = query.center_field_id
            subgraph.summary = self._summarize_subgraph(subgraph)
            results.append(subgraph)

        # Sort by timestamp (oldest first)
        results.sort(key=lambda sg: sg.timestamp)
        return results

    def extract_consolidation_batch(self, max_samples: int = 256) -> list[KGSubgraph]:
        """
        Return the oldest matured episodic subgraphs for semantic consolidation.

        Maturity criteria:
            1. Subgraph age > consolidation_frequency_minutes (configurable)
            2. At least 2 repeat observations (repeat_count >= 2)
            3. Has at least one edge (non-trivial subgraph)

        Args:
            max_samples: Maximum subgraphs to return.

        Returns:
            list[KGSubgraph] — Matured subgraphs ready for semantic training.
        """
        now = datetime.now()
        min_age = timedelta(minutes=self.config.checkpoint_interval_minutes)

        matured = []
        for sg in self.pending_consolidation:
            age = now - sg.timestamp
            if age >= min_age and sg.repeat_count >= 2 and sg.num_edges > 0:
                matured.append(sg)

        # Return oldest first (up to max_samples)
        matured.sort(key=lambda sg: sg.timestamp)
        return matured[:max_samples]

    def prewarm_semantic_query(self, pattern_embedding: torch.Tensor, k: int = 5) -> list[KGSubgraph]:
        """
        Semantic → Episodic: find K most similar episodic subgraphs.

        Uses cosine similarity between pattern_embedding and precomputed
        subgraph embeddings (computed during consolidation).

        Args:
            pattern_embedding: (pattern_embed_dim,) — Query embedding from semantic layer.
            k: Number of nearest neighbors to return.

        Returns:
            list[KGSubgraph] — Top-K most similar episodic subgraphs.
        """
        if self._subgraph_embeddings is None or len(self.pending_consolidation) == 0:
            return []

        # Normalize query embedding
        query_norm = pattern_embedding / (pattern_embedding.norm(dim=-1, keepdim=True) + 1e-8)
        query_norm = query_norm.unsqueeze(0)  # (1, D)

        # Cosine similarity                                                       # (M, D) @ (D, 1) = (M,)
        similarities = (self._subgraph_embeddings @ query_norm.T).squeeze(-1)

        # Top-K indices
        k = min(k, similarities.shape[0])
        top_k_indices = similarities.topk(k).indices.tolist()

        return [self.pending_consolidation[i] for i in top_k_indices]

    def to_subgraph_batch(self, subgraphs: list[KGSubgraph]) -> SubgraphBatch:
        """Convert KG subgraphs to batched tensor format for the semantic ML layer.

        Args:
            subgraphs: List of KGSubgraph objects (variable sizes).

        Returns:
            SubgraphBatch with padded tensor representations.
        """
        return SubgraphBatch(subgraphs)

    # ─── Public Utility Methods ───────────────────────────────────────────────

    def update_subgraph_embeddings(self, embeddings: list[torch.Tensor]) -> None:
        """Update cached subgraph embeddings for semantic-prior retrieval.

        Called after consolidation to refresh the embedding cache.

        Args:
            embeddings: List of (pattern_embed_dim,) tensors, one per
                       pending_consolidation entry.
        """
        if embeddings:
            self._subgraph_embeddings = torch.stack(embeddings)           # (M, D)
        else:
            self._subgraph_embeddings = None

    def clear_pending_consolidation(self) -> None:
        """Clear the consolidation buffer after successful consolidation."""
        self.pending_consolidation.clear()
        self._subgraph_embeddings = None

    def get_statistics(self) -> dict:
        """Return KG usage statistics."""
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "pending_consolidation": len(self.pending_consolidation),
            "writes": self._write_count,
            "queries": self._query_count,
        }

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _event_to_node_and_relations(self, event: Any) -> tuple[KGNode, list[tuple[str, str, datetime]]]:
        """Map a domain event to a KG node and its relations.

        Returns:
            tuple of (KGNode, list[(target_id, relation_type, timestamp)]).
        """
        now = datetime.now()
        node_id = f"{type(event).__name__.lower()}_{id(event)}_{now.timestamp()}"

        if isinstance(event, DiseaseEvent):
            node = KGNode(
                node_id=node_id,
                node_type="disease_event",
                timestamp=datetime.fromisoformat(event.first_observed),
                spatial_x=self._field_to_coords(event.field_id)[0],
                spatial_y=self._field_to_coords(event.field_id)[1],
                attributes={
                    "event_id": event.event_id,
                    "field_id": event.field_id,
                    "crop_cycle_id": event.crop_cycle_id,
                    "disease_name": event.disease_name,
                    "status": event.status.value,
                    "severity": event.severity,
                    "affected_area_m2": event.affected_area_m2,
                    "symptoms": event.symptoms,
                    "summary": f"Disease: {event.disease_name} in {event.field_id} (severity: {event.severity:.2f})",
                },
            )
            relations = [("field_" + event.field_id, "occurred_in", node.timestamp)]

        elif isinstance(event, TreatmentAction):
            node = KGNode(
                node_id=node_id,
                node_type="treatment",
                timestamp=datetime.fromisoformat(event.application_date),
                attributes={
                    "treatment_id": event.treatment_id,
                    "disease_event_id": event.disease_event_id,
                    "treatment_type": event.treatment_type.value,
                    "agent": event.agent,
                    "dosage": event.dosage,
                    "effectiveness": event.effectiveness,
                    "summary": f"Treatment: {event.agent} ({event.treatment_type.value})",
                },
            )
            relations = [
                ("disease_" + event.disease_event_id, "treated", node.timestamp),
            ]

        elif isinstance(event, CropCycle):
            node = KGNode(
                node_id=node_id,
                node_type="crop_cycle",
                timestamp=datetime.fromisoformat(event.planting_date),
                attributes={
                    "cycle_id": event.cycle_id,
                    "field_id": event.field_id,
                    "crop_type": event.crop_type,
                    "variety": event.variety,
                    "stages": [(s.value, t) for s, t in event.stages],
                    "summary": f"Crop: {event.crop_type} {event.variety} in {event.field_id}",
                },
            )
            relations = [("field_" + event.field_id, "planted_in", node.timestamp)]

        elif isinstance(event, dict):
            # Generic dict-based event
            node_type = event.get("type", "observation")
            # [bench-fix] Spatially index dict events by field hash, matching
            # DiseaseEvent handling so observation nodes are retrievable via
            # spatial_proximity_query.
            field_id = event.get("field_id")
            coords = self._field_to_coords(field_id) if field_id else (None, None)
            node = KGNode(
                node_id=node_id,
                node_type=node_type,
                timestamp=event.get("timestamp", now),
                spatial_x=coords[0],
                spatial_y=coords[1],
                attributes={k: v for k, v in event.items() if k != "type"},
            )
            relations = event.get("relations", [])

        else:
            raise ValueError(f"Unsupported event type: {type(event)}")

        return node, relations

    def _get_temporal_neighbors(
        self,
        node_id: str,
        relation_filter: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> list[tuple[str, KGEdge]]:
        """Get temporally sorted neighbors of a node.

        Returns list of (neighbor_id, edge) tuples sorted by edge timestamp.
        """
        neighbors = []
        for edge_id in self._node_out_edges.get(node_id, []):
            edge = self.edges.get(edge_id)
            if edge is None:
                continue

            # Relation filter
            if relation_filter is not None and edge.relation != relation_filter:
                continue

            # Temporal filter
            if from_date is not None and edge.timestamp < from_date:
                continue
            if to_date is not None and edge.timestamp > to_date:
                continue

            neighbors.append((edge.target, edge))

        # Sort by timestamp (oldest first for temporal ordering)
        neighbors.sort(key=lambda x: x[1].timestamp)
        return neighbors

    def _extract_local_subgraph(self, node_id: str, radius: int = 2) -> KGSubgraph:
        """Extract a subgraph centered on `node_id` up to `radius` hops away.

        Args:
            node_id: Center node ID.
            radius: Max BFS depth from center.

        Returns:
            KGSubgraph containing all nodes and edges within the radius.
        """
        nodes: dict[str, KGNode] = {}
        edges: dict[str, KGEdge] = {}
        visited: set[str] = set()
        frontier = [(node_id, 0)]

        while frontier:
            current_id, depth = frontier.pop(0)
            if current_id in visited or depth > radius:
                continue
            visited.add(current_id)

            if current_id in self.nodes:
                nodes[current_id] = self.nodes[current_id]

            # Outgoing edges
            for edge_id in self._node_out_edges.get(current_id, []):
                edge = self.edges.get(edge_id)
                if edge is None:
                    continue
                edges[edge_id] = edge
                if edge.target not in visited:
                    frontier.append((edge.target, depth + 1))

            # Incoming edges
            for edge_id in self._node_in_edges.get(current_id, []):
                edge = self.edges.get(edge_id)
                if edge is None:
                    continue
                edges[edge_id] = edge
                if edge.source not in visited:
                    frontier.append((edge.source, depth + 1))

        subgraph = KGSubgraph(
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            root_node_id=node_id,
            query_type="subgraph",
            timestamp=self.nodes[node_id].timestamp if node_id in self.nodes else datetime.now(),
            metadata={"radius": radius, "n_hops": depth},
        )
        # [bench-fix] Populate summary like temporal/spatial query results so
        # controller provenance and prompts carry node content, not a
        # "Subgraph at <timestamp>" placeholder.
        subgraph.summary = self._summarize_subgraph(subgraph)

        # Populate tensor representations for ML layer
        self._populate_subgraph_tensors(subgraph)
        return subgraph

    def _extract_subgraph_between(self, src_id: str, tgt_id: str, edge: KGEdge) -> KGSubgraph:
        """Extract the minimal subgraph connecting two nodes via an edge.

        Args:
            src_id: Source node ID.
            tgt_id: Target node ID.
            edge: The connecting edge.

        Returns:
            KGSubgraph containing both nodes and the edge.
        """
        nodes = []
        if src_id in self.nodes:
            nodes.append(self.nodes[src_id])
        if tgt_id in self.nodes:
            nodes.append(self.nodes[tgt_id])

        subgraph = KGSubgraph(
            nodes=nodes,
            edges=[edge],
            root_node_id=src_id,
            query_type="temporal_path",
            timestamp=edge.timestamp,
            metadata={"hop_type": edge.relation},
        )
        self._populate_subgraph_tensors(subgraph)
        return subgraph

    def _populate_subgraph_tensors(self, subgraph: KGSubgraph) -> None:
        """Populate tensor representations on a KGSubgraph for ML processing.

        Creates node_features, edge_index, and edge_features tensors.

        Shape conventions:
            node_features: (N, node_embed_dim)
            edge_index:    (2, E)
            edge_features: (E, edge_embed_dim)
        """
        N = subgraph.num_nodes
        E = subgraph.num_edges

        if N == 0:
            return

        # Build node ID → index mapping
        node_to_idx = {n.node_id: i for i, n in enumerate(subgraph.nodes)}

        # Node features: one-hot type encoding + attribute vector
        node_feat_dim = self.config.node_embed_dim
        node_feats = torch.zeros(N, node_feat_dim, dtype=torch.float32)

        type_names = sorted({n.node_type for n in subgraph.nodes})
        type_to_idx = {t: i for i, t in enumerate(type_names)}

        for i, node in enumerate(subgraph.nodes):
            # Type one-hot (scaled down to fit in embed dim)
            type_idx = type_to_idx.get(node.node_type, 0)
            if type_idx < node_feat_dim:
                node_feats[i, type_idx] = 1.0

            # Severity/numerical attributes
            severity = node.attributes.get("severity", 0.0)
            if isinstance(severity, (int, float)):
                node_feats[i, -1] = float(severity)

            # Time feature: hours since epoch / max_horizon
            hours = node.timestamp.timestamp() / 3600
            max_horizon = self.config.max_temporal_query_horizon_days * 24
            node_feats[i, -2] = hours / max_horizon if max_horizon > 0 else 0.0

        subgraph.node_features = node_feats  # (N, D_node)

        # Edge index and features
        if E > 0:
            edge_feat_dim = self.config.edge_embed_dim
            edge_idx = torch.zeros(2, E, dtype=torch.long)                    # (2, E)
            edge_feats = torch.zeros(E, edge_feat_dim, dtype=torch.float32)   # (E, D_edge)

            for j, edge in enumerate(subgraph.edges):
                src_idx = node_to_idx.get(edge.source, 0)
                tgt_idx = node_to_idx.get(edge.target, 0)
                edge_idx[0, j] = src_idx
                edge_idx[1, j] = tgt_idx

                # Encode relation type (first few dims)
                rel_hash = hash(edge.relation) % (edge_feat_dim - 1)
                edge_feats[j, rel_hash] = 1.0

                # Encode temporal delta (hours)
                if subgraph.nodes and src_idx < len(subgraph.nodes):
                    delta_hours = (edge.timestamp - subgraph.nodes[src_idx].timestamp).total_seconds() / 3600
                    edge_feats[j, -1] = delta_hours

            subgraph.edge_index = edge_idx                                     # (2, E)
            subgraph.edge_features = edge_feats                                # (E, D_edge)
        else:
            subgraph.edge_index = torch.zeros(2, 0, dtype=torch.long)
            subgraph.edge_features = torch.zeros(0, self.config.edge_embed_dim, dtype=torch.float32)

    def _merge_event(self, existing_id: str, event: Any) -> None:
        """Merge a new event into an existing node (update attributes)."""
        node = self.nodes.get(existing_id)
        if node is None:
            return

        if isinstance(event, DiseaseEvent):
            # Update severity if higher
            current_severity = node.attributes.get("severity", 0.0)
            if event.severity > current_severity:
                # Frozen dataclass — cannot modify in place; create new node
                pass  # In production, use a mutable store

        # Increment repeat count in pending consolidation
        for sg in self.pending_consolidation:
            if sg.root_node_id == existing_id:
                sg.metadata["repeat_count"] = sg.metadata.get("repeat_count", 1) + 1
                break

    def _get_field_coords(self, field_id: str) -> tuple[float, float]:
        """Look up spatial coordinates for a field ID.

        Looks for an explicit field node; falls back to the same
        deterministic hash used at ingest time (_field_to_coords) so
        query centers align with ingested events.
        """
        for node in self.nodes.values():
            if node.node_type == "field" and node.attributes.get("field_id") == field_id:
                return (node.spatial_x or 0.0, node.spatial_y or 0.0)
        return self._field_to_coords(field_id)

    def _field_to_coords(self, field_id: str) -> tuple[float, float]:
        """Generate deterministic coordinates from field ID for demo/fallback."""
        # Use hash to generate plausible coordinates
        h = hash(field_id) % (10 ** 6)
        x = (h % 1000) * 100.0       # 0-99.9 km easting
        y = ((h // 1000) % 1000) * 100.0  # 0-99.9 km northing
        return (x, y)

    def _summarize_subgraph(self, subgraph: KGSubgraph) -> str:
        """Generate a human-readable summary of a subgraph for LLM consumption."""
        parts = []
        for node in subgraph.nodes:
            summary = node.attributes.get("summary", f"{node.node_type}:{node.node_id}")
            parts.append(summary)

        for edge in subgraph.edges:
            parts.append(f"--[{edge.relation}]-->")

        return " | ".join(parts) if parts else "empty subgraph"

    # ─── nn.Module forward (for compatibility) ────────────────────────────────

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass not used directly — KG operations are call-based.

        Provided for nn.Module API compatibility. Use fast_write() and query
        methods directly.
        """
        return None

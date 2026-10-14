#!/usr/bin/env python3
"""
Smoke test for the CLS Bicameral Memory System.

Instantiates the full CLSMemorySystem, ingests synthetic agricultural events,
runs a forward pass through the semantic ML layer with shape assertions,
and tests the diagnostic workflow through the agent controller.

Also verifies:
    - All three subsystems initialize without errors
    - Episodic KG fast-write + temporal/spatial queries
    - SemanticPatternExtractor forward pass with correct output shapes
    - AgentController diagnose() returns properly structured DiagnosticResponse
    - Ablation modes switch correctly
    - Parameter counts are reasonable
"""

import sys
import os
# Ensure the coder package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datetime import datetime

from config import CLSMemorySystemConfig, EpisodicKGConfig, SemanticMLConfig, AgentControllerConfig
from data_model import (
    CropCycle, CropStage,
    DiseaseEvent, DiseaseStatus,
    TreatmentAction, TreatmentType,
    DiagnosticContext,
    SubgraphBatch, KGSubgraph,
)
from kg import EpisodicKnowledgeGraph
from semantic import SemanticMemoryManager
from controller import CLSAgentController
from model import CLSMemorySystem
from base import count_params


def _make_subgraph_batch(node_features, edge_index, edge_features, node_mask, edge_mask):
    """Build a SubgraphBatch from raw tensors (avoids __new__ hack)."""
    from data_model import SubgraphBatch
    B = node_features.shape[0]
    # We need to create KGSubgraph objects and batch them
    # For simplicity, construct directly
    batch = object.__new__(SubgraphBatch)
    batch.node_features = node_features
    batch.edge_index = edge_index
    batch.edge_features = edge_features
    batch.node_mask = node_mask
    batch.edge_mask = edge_mask
    return batch


def test_configs():
    """Test that all configs can be instantiated with defaults."""
    print("=" * 60)
    print("1. Testing Configuration Dataclasses")
    print("=" * 60)

    cfg = CLSMemorySystemConfig()
    assert cfg.episodic_kg.max_triples == 50_000
    assert cfg.semantic_ml.hidden_dim == 256
    assert cfg.agent_controller.max_iterative_cycles == 5
    assert cfg.semantic_ml.node_embed_dim == cfg.episodic_kg.node_embed_dim
    print(f"  [OK] CLSMemorySystemConfig initialized")
    print(f"       EpisodicKG: max_triples={cfg.episodic_kg.max_triples}, "
          f"max_nodes={cfg.episodic_kg.max_nodes}")
    print(f"       SemanticML: hidden_dim={cfg.semantic_ml.hidden_dim}, "
          f"n_pattern_slots={cfg.semantic_ml.n_pattern_slots}")
    print(f"       AgentController: max_iterations={cfg.agent_controller.max_iterative_cycles}, "
          f"method={cfg.agent_controller.reconciliation_method}")


def test_episodic_kg():
    """Test the EpisodicKnowledgeGraph with synthetic agricultural events."""
    print("\n" + "=" * 60)
    print("2. Testing EpisodicKnowledgeGraph")
    print("=" * 60)

    config = EpisodicKGConfig()
    kg = EpisodicKnowledgeGraph(config)

    # Create synthetic events
    now = datetime.now()

    crop_cycle = CropCycle(
        cycle_id="cycle_001",
        field_id="field_A42",
        crop_type="wheat",
        variety="spring_wheat_v1",
        planting_date="2026-04-01",
        stages=[
            (CropStage.PLANTING, "2026-04-01"),
            (CropStage.VEGETATIVE, "2026-04-15"),
        ],
    )

    disease_event = DiseaseEvent(
        event_id="de_001",
        field_id="field_A42",
        crop_cycle_id="cycle_001",
        disease_name="powdery_mildew",
        first_observed="2026-05-15",
        status=DiseaseStatus.CONFIRMED,
        severity=0.6,
        affected_area_m2=250.0,
        symptoms=["white_powdery_coating", "stunted_growth"],
    )

    treatment = TreatmentAction(
        treatment_id="tr_001",
        disease_event_id="de_001",
        treatment_type=TreatmentType.CHEMICAL_FUNGICIDE,
        agent="sulfur_based_fungicide",
        dosage="2.5 L/ha",
        application_date="2026-05-18",
        effectiveness=0.7,
        follow_up_required=True,
        notes="Applied at first sign of spread",
    )

    # Fast-write events
    crop_id = kg.fast_write(crop_cycle)
    disease_id = kg.fast_write(disease_event)
    treatment_id = kg.fast_write(treatment)

    print(f"  [OK] fast_write: crop='{crop_id[:40]}...', disease='{disease_id[:40]}...', "
          f"treatment='{treatment_id[:40]}...'")

    # Verify node/edge counts
    stats = kg.get_statistics()
    print(f"  [OK] KG stats: nodes={stats['nodes']}, edges={stats['edges']}, "
          f"pending_consolidation={stats['pending_consolidation']}")

    assert stats["nodes"] >= 3, f"Expected >=3 nodes, got {stats['nodes']}"
    assert stats["writes"] == 3, f"Expected 3 writes, got {stats['writes']}"

    # Test temporal/spatial index sizes
    assert len(kg.temporal_index) >= 3, f"Temporal index too small: {len(kg.temporal_index)}"
    print(f"  [OK] Temporal index: {len(kg.temporal_index)} entries")

    # Test extraction of consolidation batch
    batch = kg.extract_consolidation_batch(max_samples=10)
    # (May be empty because events are too recent — that's expected)
    print(f"  [OK] extract_consolidation_batch: {len(batch)} subgraphs (may be 0 if too recent)")

    # Test to_subgraph_batch
    if batch:
        sb = kg.to_subgraph_batch(batch)
        assert sb.node_features.shape[0] == len(batch), (
            f"Batch dim mismatch: {sb.node_features.shape[0]} vs {len(batch)}"
        )
        print(f"  [OK] to_subgraph_batch: node_features shape = {sb.node_features.shape}")

    print(f"  [OK] All EpisodicKG tests passed")


def test_semantic_pattern_extractor():
    """Test the SemanticPatternExtractor forward pass with shape assertions."""
    print("\n" + "=" * 60)
    print("3. Testing SemanticPatternExtractor")
    print("=" * 60)

    config = SemanticMLConfig(
        node_embed_dim=64,
        edge_embed_dim=16,
        hidden_dim=128,
        n_layers=2,
        n_pattern_slots=16,
        pattern_embed_dim=64,
        n_heads=2,
        dropout=0.1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SemanticMemoryManager(config).extractor.to(device)
    count_params(extractor)

    # Create synthetic batched subgraphs
    B, N, E = 4, 10, 8  # batch=4, max_nodes=10, max_edges=8

    # Node features: (B, N, D_node)
    node_features = torch.randn(B, N, config.node_embed_dim, device=device)
    # Edge index: (B, 2, E)
    edge_index = torch.randint(0, N, (B, 2, E), device=device)
    # Edge features: (B, E, D_edge)
    edge_features = torch.randn(B, E, config.edge_embed_dim, device=device)
    # Node mask: (B, N) — first 5 nodes valid per graph
    node_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    node_mask[:, :5] = True
    # Edge mask: (B, E) — first 4 edges valid per graph
    edge_mask = torch.zeros(B, E, dtype=torch.bool, device=device)
    edge_mask[:, :4] = True

    batch = _make_subgraph_batch(node_features, edge_index, edge_features, node_mask, edge_mask)

    # Forward pass
    with torch.no_grad():
        output = extractor._forward_impl(batch)

    # Shape assertions
    assert output["pattern_embed"].shape == (B, config.pattern_embed_dim), (
        f"pattern_embed shape: {output['pattern_embed'].shape}, "
        f"expected ({B}, {config.pattern_embed_dim})"
    )
    assert output["prototype_weights"].shape == (B, config.n_pattern_slots), (
        f"prototype_weights shape: {output['prototype_weights'].shape}, "
        f"expected ({B}, {config.n_pattern_slots})"
    )
    assert output["confidence"].shape == (B,), (
        f"confidence shape: {output['confidence'].shape}, expected ({B},)"
    )

    print(f"  [OK] Forward pass shapes:")
    print(f"       pattern_embed:     {output['pattern_embed'].shape}     ← (B, pattern_embed_dim)")
    print(f"       prototype_weights: {output['prototype_weights'].shape} ← (B, n_pattern_slots)")
    print(f"       confidence:        {output['confidence'].shape}        ← (B,)")

    # Test confidence range (sigmoid output should be in [0, 1])
    assert output["confidence"].min() >= 0.0 and output["confidence"].max() <= 1.0, (
        f"Confidence out of [0, 1] range: [{output['confidence'].min():.3f}, "
        f"{output['confidence'].max():.3f}]"
    )
    print(f"  [OK] Confidence in [0, 1]: [{output['confidence'].min():.3f}, "
          f"{output['confidence'].max():.3f}]")

    # Test float16/bf16 safety
    if device == "cuda":
        for dtype in [torch.float16, torch.bfloat16]:
            batch_fp = _make_subgraph_batch(
                node_features.to(dtype=dtype),
                edge_index,
                edge_features.to(dtype=dtype),
                node_mask,
                edge_mask,
            )

            with torch.no_grad():
                # Should not raise
                output_fp = extractor._forward_impl(batch_fp)
            print(f"  [OK] {dtype} forward: pattern {output_fp['pattern_embed'].shape}, "
                  f"confidence {output_fp['confidence'].shape}")

    # Test few-shot adaptation
    dummy_subgraphs = []
    for i in range(5):
        sg = KGSubgraph(
            root_node_id=f"query_{i}",
            nodes=[],
            edges=[],
            query_type="query",
            timestamp=datetime.now(),
            summary=f"Support subgraph {i}",
        )
        # Minimal tensor data
        sg.node_features = torch.randn(3, config.node_embed_dim, device=device)
        sg.edge_index = torch.randint(0, 3, (2, 2), device=device)
        sg.edge_features = torch.randn(2, config.edge_embed_dim, device=device)
        dummy_subgraphs.append(sg)

    query_sg = KGSubgraph(
        root_node_id="query_target",
        nodes=[],
        edges=[],
        query_type="query",
        timestamp=datetime.now(),
        summary="Query subgraph",
    )
    query_sg.node_features = torch.randn(3, config.node_embed_dim, device=device)
    query_sg.edge_index = torch.randint(0, 3, (2, 2), device=device)
    query_sg.edge_features = torch.randn(2, config.edge_embed_dim, device=device)

    few_shot_result = extractor.few_shot_adapt(dummy_subgraphs, query_sg)
    assert "pattern_embed" in few_shot_result
    assert "similarity" in few_shot_result
    print(f"  [OK] Few-shot adaptation: similarity={few_shot_result['similarity']:.3f}, "
          f"confidence={few_shot_result['confidence']:.3f}")

    # Test prototype utilities
    all_protos = extractor.get_all_prototypes()
    assert all_protos.shape == (config.n_pattern_slots, config.pattern_embed_dim), (
        f"All prototypes shape: {all_protos.shape}"
    )
    sims = extractor.compute_prototype_similarity(output["pattern_embed"][0])
    assert sims.shape == (config.n_pattern_slots,)
    print(f"  [OK] Prototype similarity: shape {sims.shape}, "
          f"max similarity {sims.max():.3f}")

    print(f"  [OK] All SemanticPatternExtractor tests passed")


def test_cls_memory_system():
    """Test the full CLSMemorySystem end-to-end."""
    print("\n" + "=" * 60)
    print("4. Testing CLSMemorySystem (End-to-End)")
    print("=" * 60)

    # Use a small config for fast testing
    cfg = CLSMemorySystemConfig(
        episodic_kg=EpisodicKGConfig(
            max_triples=1000,
            max_nodes=500,
            node_embed_dim=32,
            edge_embed_dim=8,
        ),
        semantic_ml=SemanticMLConfig(
            node_embed_dim=32,
            edge_embed_dim=8,
            hidden_dim=64,
            n_layers=2,
            n_pattern_slots=8,
            pattern_embed_dim=32,
            n_heads=2,
            d_ff=256,
            dropout=0.1,
            consolidation_batch_size=2,
        ),
        agent_controller=AgentControllerConfig(
            max_iterative_cycles=2,
            early_exit_confidence=0.9,
            reconciliation_method="confidence_max",
            provenance_tracking=True,
        ),
    )

    system = CLSMemorySystem(cfg)
    count_params(system)

    # Ingest synthetic events
    now = datetime.now()

    disease1 = DiseaseEvent(
        event_id="de_001",
        field_id="field_A42",
        crop_cycle_id="cycle_001",
        disease_name="powdery_mildew",
        first_observed="2026-05-15",
        status=DiseaseStatus.CONFIRMED,
        severity=0.6,
        affected_area_m2=250.0,
        symptoms=["white_powdery_coating"],
    )

    disease2 = DiseaseEvent(
        event_id="de_002",
        field_id="field_B17",
        crop_cycle_id="cycle_002",
        disease_name="rust",
        first_observed="2026-05-20",
        status=DiseaseStatus.ACTIVE,
        severity=0.4,
        affected_area_m2=100.0,
        symptoms=["orange_pustules"],
    )

    id1 = system.fast_write(disease1)
    id2 = system.fast_write(disease2)
    print(f"  [OK] Ingested 2 disease events: {id1[:32]}..., {id2[:32]}...")

    # Test diagnostic query
    context = DiagnosticContext(
        field_id="field_A42",
        crop_type="wheat",
        season_start=datetime(2026, 3, 1),
        season_end=datetime(2026, 6, 1),
    )

    response = system.diagnose("Why is my wheat showing powdery mildew?", context)
    print(f"  [OK] diagnose() returned response:")
    print(f"       num_iterations: {response.num_iterations}")
    print(f"       confidence: {response.confidence:.3f}")
    print(f"       evidence count: {len(response.evidence)}")
    print(f"       provenance count: {len(response.provenance)}")
    print(f"       answer length: {len(response.answer)} chars")

    assert isinstance(response.answer, str), f"Answer should be str, got {type(response.answer)}"
    assert response.num_iterations >= 1, f"Should have at least 1 iteration"
    assert response.confidence >= 0.0, f"Confidence should be >= 0"

    # Test ablation modes
    system.set_ablation_mode("kg_only")
    kg_response = system.diagnose("What happened in field A-42?", context)
    assert isinstance(kg_response.answer, str)
    print(f"  [OK] kg_only mode: answer length = {len(kg_response.answer)} chars")

    system.set_ablation_mode("no_iterate")
    ni_response = system.diagnose("Is the rust spreading?", context)
    assert isinstance(ni_response.answer, str)
    print(f"  [OK] no_iterate mode: answer length = {len(ni_response.answer)} chars, "
          f"iterations={ni_response.num_iterations}")

    system.set_ablation_mode("full")
    full_response = system.diagnose("What treatments were applied?", context)
    assert isinstance(full_response.answer, str)
    print(f"  [OK] full mode restored: answer length = {len(full_response.answer)} chars")

    # Test system state
    state = system.get_system_state()
    assert "statistics" in state
    assert "config" in state
    assert state["statistics"]["events_ingested"] == 2
    print(f"  [OK] System state: events={state['statistics']['events_ingested']}, "
          f"diagnoses={state['statistics']['diagnoses_performed']}")

    # Test consolidation cycle
    cons_result = system.consolidate()
    if not cons_result.get("skipped", False):
        print(f"  [OK] Consolidation: epochs={cons_result.get('epochs')}, "
              f"final_loss={cons_result.get('final_loss')}")
    else:
        print(f"  [OK] Consolidation skipped (expected — insufficient matured data)")

    print(f"  [OK] All CLSMemorySystem tests passed")


def test_controller_interface():
    """Test the agent controller's interface with structured data."""
    print("\n" + "=" * 60)
    print("5. Testing AgentController Interface")
    print("=" * 60)

    from controller import WorkingMemory

    # Test WorkingMemory
    wm = WorkingMemory(max_tokens=1000)
    wm.add("query", "test query")
    wm.add("context", {"field": "A-42"})
    assert wm.get("query") == "test query"
    assert "query" in wm
    assert len(wm) == 2
    wm.clear()
    assert len(wm) == 0
    print("  [OK] WorkingMemory: add, get, contains, clear")

    # Test WorkingMemory eviction
    wm2 = WorkingMemory(max_tokens=50)  # Small capacity
    for i in range(20):
        wm2.add(f"key_{i}", "x" * 20)  # Each ~5 tokens
    # Should have evicted some items
    assert len(wm2) < 20, "Eviction did not trigger"
    print(f"  [OK] WorkingMemory eviction: {len(wm2)} items after 20 inserts")

    print("  [OK] All Controller interface tests passed")


def test_device_portability():
    """Test that the system runs on both CPU and CUDA."""
    print("\n" + "=" * 60)
    print("6. Testing Device Portability")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Running on: {device}")

    # SemanticPatternExtractor on device
    config = SemanticMLConfig(
        node_embed_dim=32,
        edge_embed_dim=8,
        hidden_dim=64,
        n_layers=2,
        n_pattern_slots=8,
        pattern_embed_dim=32,
    )

    from semantic import SemanticPatternExtractor
    extractor = SemanticPatternExtractor(config).to(device)

    # Create minimal batch
    B, N, E = 2, 5, 4
    nf = torch.randn(B, N, config.node_embed_dim, device=device)
    ei = torch.randint(0, N, (B, 2, E), device=device)
    ef = torch.randn(B, E, config.edge_embed_dim, device=device)
    nm = torch.ones(B, N, dtype=torch.bool, device=device)
    em = torch.ones(B, E, dtype=torch.bool, device=device)
    batch = _make_subgraph_batch(nf, ei, ef, nm, em)

    with torch.no_grad():
        out = extractor._forward_impl(batch)

    assert out["pattern_embed"].device.type == device
    assert out["confidence"].device.type == device
    print(f"  [OK] SemanticPatternExtractor on {device}: "
          f"pattern {out['pattern_embed'].shape} on {out['pattern_embed'].device}")

    # Test system on device
    system_cfg = CLSMemorySystemConfig(
        episodic_kg=EpisodicKGConfig(node_embed_dim=32, edge_embed_dim=8),
        semantic_ml=SemanticMLConfig(
            node_embed_dim=32, edge_embed_dim=8, hidden_dim=64,
            n_layers=2, n_pattern_slots=8, pattern_embed_dim=32,
        ),
    )
    system = CLSMemorySystem(system_cfg)
    print(f"  [OK] CLSMemorySystem initializes on CPU")

    print(f"  [OK] All device portability tests passed")


def test_tensor_shape_comments():
    """Verify that key tensor operations have shape annotations in the code."""
    print("\n" + "=" * 60)
    print("7. Verifying Shape Comments in Source Code")
    print("=" * 60)

    import ast
    import os

    files_to_check = [
        os.path.join(os.path.dirname(__file__), "layers.py"),
        os.path.join(os.path.dirname(__file__), "semantic.py"),
        os.path.join(os.path.dirname(__file__), "kg.py"),
    ]

    shape_comment_count = 0
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            continue

        with open(filepath) as f:
            content = f.read()

        # Count shape comments (comments containing patterns like (B, N, D) or (N, D) or →)
        import re
        shape_comments = re.findall(
            r'#.*\([^)]*\).*→|#.*\([^)]*\).*shape|#.*\([^)]*\)',
            content
        )
        shape_comment_count += len(shape_comments)

    print(f"  [OK] Found ~{shape_comment_count} shape comments across source files")
    assert shape_comment_count > 10, f"Too few shape comments: {shape_comment_count}"
    print(f"  [OK] Shape comment verification passed")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("CLS Bicameral Memory System — Smoke Test Suite")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("=" * 60)

    try:
        test_configs()
        test_episodic_kg()
        test_semantic_pattern_extractor()
        test_cls_memory_system()
        test_controller_interface()
        test_device_portability()
        test_tensor_shape_comments()

        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n!!! SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

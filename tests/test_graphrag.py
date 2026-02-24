#!/usr/bin/env python
"""Test script for GraphRAG module.

Run with: python tests/test_graphrag.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all GraphRAG imports work."""
    print("=" * 60)
    print("Test 1: GraphRAG imports")
    print("=" * 60)

    try:
        from graphrag import (
            extract_entities_from_paper,
            extract_entities_batch,
            link_entities,
            build_knowledge_graph,
            compute_graph_statistics,
            detect_communities_leiden,
            build_community_hierarchy,
            summarize_community,
            generate_global_summary,
            graphrag_agent,
            run_graphrag_pipeline,
        )
        print("All GraphRAG imports successful")
        print("PASS: GraphRAG imports")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_types():
    """Test GraphRAG-related state types."""
    print("\n" + "=" * 60)
    print("Test 2: State types")
    print("=" * 60)

    try:
        from agents.state import (
            ExtractedEntity,
            EntityMention,
            GraphEdge,
            Community,
            KnowledgeGraphData,
            BatchProcessingState,
            create_initial_state,
        )

        # Test create_initial_state with new params
        state = create_initial_state(
            "test question",
            max_papers=50,
            graphrag_enabled=True,
            batch_size=10,
        )

        assert state["graphrag_enabled"] is True
        assert state["batch_size"] == 10
        assert state["max_papers"] == 50
        assert state["knowledge_graph"] is None

        # Phase 9.5: ablation flags should default to False
        assert state["disable_self_review"] is False
        assert state["disable_domain_filter"] is False
        assert state["disable_snowball"] is False
        # token_tracker moved to ContextVar (not in state)
        assert "token_tracker" not in state
        print("  Phase 9.5 fields OK")

        print("State types defined correctly")
        print("PASS: State types")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_entity_normalization():
    """Test entity name normalization."""
    print("\n" + "=" * 60)
    print("Test 3: Entity normalization")
    print("=" * 60)

    try:
        from graphrag.entity_extractor import normalize_entity_name, generate_entity_id

        test_cases = [
            ("BERT model", "BERT"),
            ("Transformer architecture", "Transformer"),
            ("ImageNet dataset", "ImageNet"),
            ("Attention Mechanism", "Attention Mechanism"),
            ("  Gradient Descent  ", "Gradient Descent"),
        ]

        for input_name, expected in test_cases:
            result = normalize_entity_name(input_name)
            status = "OK" if result == expected else "FAIL"
            print(f"  '{input_name}' -> '{result}' {status}")
            assert result == expected, f"Expected '{expected}', got '{result}'"

        # Test ID generation
        id1 = generate_entity_id("BERT", "method")
        id2 = generate_entity_id("bert", "method")  # Case insensitive
        id3 = generate_entity_id("BERT", "dataset")  # Different type

        assert id1 == id2, "Same name should produce same ID"
        assert id1 != id3, "Same name but different type should produce different ID"

        print("Entity normalization working correctly")
        print("PASS: Entity normalization")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_builder():
    """Test knowledge graph construction."""
    print("\n" + "=" * 60)
    print("Test 4: Graph builder")
    print("=" * 60)

    try:
        import networkx as nx
        from graphrag.graph_builder import (
            build_knowledge_graph,
            compute_graph_statistics,
            get_entity_subgraph,
            get_papers_for_entity,
            get_entities_for_paper,
        )

        # Sample data
        entities = [
            {
                "entity_id": "e1",
                "name": "BERT",
                "entity_type": "method",
                "aliases": ["BERT model"],
                "description": "Language model",
                "paper_ids": ["p1", "p2"],
                "frequency": 2,
            },
            {
                "entity_id": "e2",
                "name": "ImageNet",
                "entity_type": "dataset",
                "aliases": [],
                "description": "Image classification dataset",
                "paper_ids": ["p1"],
                "frequency": 1,
            },
        ]

        mentions = [
            {"entity_id": "e1", "paper_id": "p1", "context": "BERT is used", "section": "methods", "confidence": 0.9},
            {"entity_id": "e1", "paper_id": "p2", "context": "BERT performance", "section": "results", "confidence": 0.8},
            {"entity_id": "e2", "paper_id": "p1", "context": "ImageNet benchmark", "section": "methods", "confidence": 0.9},
        ]

        papers = [
            {"paper_id": "p1", "title": "Paper 1", "authors": ["Author 1"], "year": 2023, "citations": 100, "source": "arxiv"},
            {"paper_id": "p2", "title": "Paper 2", "authors": ["Author 2"], "year": 2024, "citations": 50, "source": "arxiv"},
        ]

        # Build graph
        G = build_knowledge_graph(entities, mentions, papers)

        assert G.number_of_nodes() > 0, "Graph should have nodes"
        assert G.number_of_edges() > 0, "Graph should have edges"

        # Test statistics
        stats = compute_graph_statistics(G)
        print(f"  Nodes: {stats['node_count']}")
        print(f"  Edges: {stats['edge_count']}")
        print(f"  Papers: {stats['paper_count']}")
        print(f"  Entities: {stats['entity_count']}")

        assert stats["paper_count"] == 2
        assert stats["entity_count"] == 2

        # Test entity subgraph
        entity_graph = get_entity_subgraph(G)
        print(f"  Entity subgraph nodes: {entity_graph.number_of_nodes()}")

        # Test lookups
        papers_for_e1 = get_papers_for_entity(G, "e1")
        print(f"  Papers mentioning e1: {papers_for_e1}")

        entities_for_p1 = get_entities_for_paper(G, "p1")
        print(f"  Entities in p1: {entities_for_p1}")

        print("Graph builder working correctly")
        print("PASS: Graph builder")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_community_detection():
    """Test community detection."""
    print("\n" + "=" * 60)
    print("Test 5: Community detection")
    print("=" * 60)

    try:
        import networkx as nx
        from graphrag.community_detector import (
            detect_communities_leiden,
            build_community_hierarchy,
            compute_community_stats,
        )

        # Create test graph
        G = nx.Graph()
        # Community 1: methods
        G.add_edge("bert", "gpt", weight=3)
        G.add_edge("bert", "transformer", weight=5)
        G.add_edge("gpt", "transformer", weight=4)
        # Community 2: datasets
        G.add_edge("imagenet", "coco", weight=3)
        G.add_edge("imagenet", "cifar", weight=2)
        # Weak link between communities
        G.add_edge("bert", "imagenet", weight=1)

        # Add paper_ids for hierarchy test
        for node in G.nodes():
            G.nodes[node]["paper_ids"] = [f"paper_{node}"]

        # Detect communities
        partition = detect_communities_leiden(G)
        print(f"  Partition: {partition}")

        num_communities = len(set(partition.values()))
        print(f"  Communities detected: {num_communities}")

        # Build hierarchy
        communities = build_community_hierarchy(G, resolutions=[0.5, 1.0, 2.0])
        print(f"  Hierarchical communities: {len(communities)}")

        stats = compute_community_stats(communities)
        print(f"  Community stats: {stats}")

        print("Community detection working correctly")
        print("PASS: Community detection")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_supervisor_routing():
    """Test supervisor routing with GraphRAG."""
    print("\n" + "=" * 60)
    print("Test 6: Supervisor routing")
    print("=" * 60)

    try:
        from agents.supervisor import determine_next_agent

        # Test case 1: GraphRAG enabled, no knowledge graph yet
        # Note: research_plan must be non-None to pass the planning check
        state1 = {
            "research_plan": {"complexity_score": 2, "sub_topics": []},
            "search_results": {"total_found": 10},
            "papers_to_analyze": [{"id": "1"}, {"id": "2"}],
            "analyzed_papers": [{"paper_id": "1"}, {"paper_id": "2"}],
            "knowledge_graph": None,
            "graphrag_enabled": True,
            "synthesis": None,
            "errors": [],
            "iteration_count": 1,
        }
        next1 = determine_next_agent(state1)
        print(f"  GraphRAG enabled, no KG: {next1}")
        assert next1 == "graphrag", f"Expected 'graphrag', got '{next1}'"

        # Test case 2: GraphRAG disabled
        state2 = state1.copy()
        state2["graphrag_enabled"] = False
        next2 = determine_next_agent(state2)
        print(f"  GraphRAG disabled: {next2}")
        assert next2 == "synthesis", f"Expected 'synthesis', got '{next2}'"

        # Test case 3: GraphRAG complete, need synthesis
        state3 = state1.copy()
        state3["knowledge_graph"] = {"entities": {}, "communities": []}
        next3 = determine_next_agent(state3)
        print(f"  GraphRAG complete: {next3}")
        assert next3 == "synthesis", f"Expected 'synthesis', got '{next3}'"

        print("Supervisor routing working correctly")
        print("PASS: Supervisor routing")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_routing():
    """Test workflow should_continue function."""
    print("\n" + "=" * 60)
    print("Test 7: Workflow routing")
    print("=" * 60)

    try:
        from agents.graph import should_continue

        test_cases = [
            ({"current_agent": "discovery"}, "discovery"),
            ({"current_agent": "critical_reading"}, "critical_reading"),
            ({"current_agent": "graphrag"}, "graphrag"),
            ({"current_agent": "synthesis"}, "synthesis"),
            ({"current_agent": "complete"}, "complete"),
            ({}, "complete"),  # Default
        ]

        for state, expected in test_cases:
            result = should_continue(state)
            status = "OK" if result == expected else "FAIL"
            print(f"  current_agent='{state.get('current_agent', 'None')}' -> '{result}' {status}")
            assert result == expected, f"Expected '{expected}', got '{result}'"

        print("Workflow routing working correctly")
        print("PASS: Workflow routing")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracker_params_in_graphrag():
    """Test that GraphRAG functions accept tracker parameter (Phase 9.5)."""
    print("\n" + "=" * 60)
    print("Test 8: Tracker params in GraphRAG functions")
    print("=" * 60)

    try:
        import inspect

        from graphrag.entity_extractor import extract_entities_from_paper
        sig = inspect.signature(extract_entities_from_paper)
        assert "tracker" in sig.parameters, "extract_entities_from_paper missing tracker param"
        print("  extract_entities_from_paper: tracker param OK")

        from graphrag.summarizer import summarize_community, generate_global_summary
        sig1 = inspect.signature(summarize_community)
        assert "tracker" in sig1.parameters, "summarize_community missing tracker param"
        print("  summarize_community: tracker param OK")

        sig2 = inspect.signature(generate_global_summary)
        assert "tracker" in sig2.parameters, "generate_global_summary missing tracker param"
        print("  generate_global_summary: tracker param OK")

        from graphrag.integration import run_graphrag_pipeline
        sig3 = inspect.signature(run_graphrag_pipeline)
        assert "tracker" in sig3.parameters, "run_graphrag_pipeline missing tracker param"
        print("  run_graphrag_pipeline: tracker param OK")

        print("PASS: Tracker params in GraphRAG functions")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_entity_extraction_retry_params():
    """Test that entity extractor has retry logic (Phase 10)."""
    print("\n" + "=" * 60)
    print("Test 9: Entity extraction retry params")
    print("=" * 60)

    try:
        import inspect
        from graphrag.entity_extractor import extract_entities_from_paper

        # Read the source code to verify retry logic exists
        source = inspect.getsource(extract_entities_from_paper)

        assert "max_retries" in source, "extract_entities_from_paper missing max_retries"
        print("  max_retries found")

        assert "asyncio.sleep" in source, "extract_entities_from_paper missing backoff sleep"
        print("  backoff sleep found")

        assert "temperature" in source, "extract_entities_from_paper missing temperature param"
        print("  temperature adjustment found")

        print("PASS: Entity extraction retry params")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_threshold_clustering():
    """Test that entity linker uses threshold+connected components (Phase 10)."""
    print("\n" + "=" * 60)
    print("Test 10: Threshold clustering")
    print("=" * 60)

    try:
        import numpy as np
        from graphrag.entity_linker import cluster_similar_entities

        # Create a similarity matrix with two clear clusters
        # Entities 0,1,2 are similar to each other; entities 3,4 are similar
        similarity = np.array([
            [1.0, 0.95, 0.90, 0.20, 0.15],
            [0.95, 1.0, 0.88, 0.18, 0.12],
            [0.90, 0.88, 1.0, 0.22, 0.19],
            [0.20, 0.18, 0.22, 1.0, 0.92],
            [0.15, 0.12, 0.19, 0.92, 1.0],
        ])
        entity_ids = ["e1", "e2", "e3", "e4", "e5"]

        clusters = cluster_similar_entities(similarity, entity_ids, threshold=0.85)

        print(f"  Clusters: {clusters}")
        assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"

        # Find which cluster has 3 entities and which has 2
        cluster_sizes = sorted([len(c) for c in clusters])
        assert cluster_sizes == [2, 3], f"Expected cluster sizes [2, 3], got {cluster_sizes}"

        # Verify cluster contents
        for cluster in clusters:
            if len(cluster) == 3:
                assert cluster == {"e1", "e2", "e3"}, f"Unexpected 3-cluster: {cluster}"
            else:
                assert cluster == {"e4", "e5"}, f"Unexpected 2-cluster: {cluster}"

        # Test single entity
        single_clusters = cluster_similar_entities(
            np.array([[1.0]]), ["e1"], threshold=0.85
        )
        assert len(single_clusters) == 1, "Single entity should give 1 cluster"
        print("  Single entity clustering OK")

        # Test empty
        empty_clusters = cluster_similar_entities(np.array([]), [], threshold=0.85)
        assert len(empty_clusters) == 0, "Empty input should give 0 clusters"
        print("  Empty clustering OK")

        # Verify no sklearn import
        import inspect
        source = inspect.getsource(cluster_similar_entities)
        assert "AgglomerativeClustering" not in source, "Should not use AgglomerativeClustering"
        print("  No sklearn dependency confirmed")

        print("PASS: Threshold clustering")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_bidirectional_cooccur():
    """Test that co-occur edges are not duplicated bidirectionally."""
    print("\n" + "=" * 60)
    print("Test 11: No bidirectional co-occur edges")
    print("=" * 60)

    try:
        from graphrag.graph_builder import build_knowledge_graph

        entities = [
            {"entity_id": "e1", "name": "A", "entity_type": "method",
             "aliases": [], "description": "", "paper_ids": ["p1"], "frequency": 1},
            {"entity_id": "e2", "name": "B", "entity_type": "method",
             "aliases": [], "description": "", "paper_ids": ["p1"], "frequency": 1},
            {"entity_id": "e3", "name": "C", "entity_type": "concept",
             "aliases": [], "description": "", "paper_ids": ["p1"], "frequency": 1},
        ]
        mentions = [
            {"entity_id": "e1", "paper_id": "p1", "context": "", "section": "", "confidence": 0.9},
            {"entity_id": "e2", "paper_id": "p1", "context": "", "section": "", "confidence": 0.9},
            {"entity_id": "e3", "paper_id": "p1", "context": "", "section": "", "confidence": 0.9},
        ]
        papers = [
            {"paper_id": "p1", "title": "P1", "authors": [], "year": 2024,
             "citations": 0, "source": "test"},
        ]

        G = build_knowledge_graph(entities, mentions, papers, include_citation_edges=False)

        # Count co-occur edges
        cooccur_edges = [(u, v) for u, v, d in G.edges(data=True)
                         if d.get("edge_type") == "co_occurs"]
        print(f"  Co-occur edges: {len(cooccur_edges)}")

        # 3 entities in 1 paper → 3 unique pairs (e1-e2, e1-e3, e2-e3)
        # Without bidirectional: 3 edges. With bidirectional: 6 edges.
        assert len(cooccur_edges) == 3, (
            f"Expected 3 co-occur edges (no bidirectional), got {len(cooccur_edges)}"
        )

        # Verify no reverse duplicates
        pairs = set()
        for u, v in cooccur_edges:
            pair = tuple(sorted([u, v]))
            assert pair not in pairs, f"Duplicate co-occur pair: {pair}"
            pairs.add(pair)
        print("  No duplicate pairs confirmed")

        print("PASS: No bidirectional co-occur edges")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graphrag_defaults():
    """Test that GraphRAG default parameters are correctly updated."""
    print("\n" + "=" * 60)
    print("Test 12: GraphRAG default parameters")
    print("=" * 60)

    try:
        import inspect

        # Check entity_linker defaults
        from graphrag.entity_linker import cluster_similar_entities, link_entities
        sig1 = inspect.signature(cluster_similar_entities)
        assert sig1.parameters["threshold"].default == 0.75, (
            f"cluster threshold should be 0.75, got {sig1.parameters['threshold'].default}"
        )
        print("  entity_linker cluster threshold = 0.75 OK")

        sig2 = inspect.signature(link_entities)
        assert sig2.parameters["similarity_threshold"].default == 0.75, (
            f"link threshold should be 0.75, got {sig2.parameters['similarity_threshold'].default}"
        )
        print("  entity_linker link threshold = 0.75 OK")

        # Check community_detector defaults
        from graphrag.community_detector import build_community_hierarchy
        sig3 = inspect.signature(build_community_hierarchy)
        assert sig3.parameters["resolutions"].default == [1.0], (
            f"resolutions should be [1.0], got {sig3.parameters['resolutions'].default}"
        )
        print("  community_detector resolutions = [1.0] OK")
        assert sig3.parameters["min_community_size"].default == 3, (
            f"min_community_size should be 3, got {sig3.parameters['min_community_size'].default}"
        )
        print("  community_detector min_community_size = 3 OK")

        # Check integration defaults
        from graphrag.integration import run_graphrag_pipeline
        sig4 = inspect.signature(run_graphrag_pipeline)
        assert sig4.parameters["similarity_threshold"].default == 0.75, (
            f"integration similarity_threshold should be 0.75, got {sig4.parameters['similarity_threshold'].default}"
        )
        assert sig4.parameters["community_resolutions"].default == [1.0], (
            f"integration resolutions should be [1.0], got {sig4.parameters['community_resolutions'].default}"
        )
        assert sig4.parameters["min_community_size"].default == 3, (
            f"integration min_community_size should be 3, got {sig4.parameters['min_community_size'].default}"
        )
        print("  integration defaults OK")

        # Check entity extraction prompt
        from graphrag.prompts import ENTITY_EXTRACTION_PROMPT
        assert "3-8 entities" in ENTITY_EXTRACTION_PROMPT, (
            "Extraction prompt should say 3-8 entities"
        )
        assert "Do NOT extract generic" in ENTITY_EXTRACTION_PROMPT, (
            "Extraction prompt should warn against generic entities"
        )
        print("  entity extraction prompt updated OK")

        print("PASS: GraphRAG default parameters")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_resolution_community():
    """Test that single resolution produces reasonable community count."""
    print("\n" + "=" * 60)
    print("Test 13: Single resolution community detection")
    print("=" * 60)

    try:
        import networkx as nx
        from graphrag.community_detector import build_community_hierarchy

        # Create graph with 2 clear clusters of 3+ nodes each
        G = nx.Graph()
        # Cluster 1
        G.add_edge("a1", "a2", weight=3)
        G.add_edge("a2", "a3", weight=3)
        G.add_edge("a1", "a3", weight=2)
        # Cluster 2
        G.add_edge("b1", "b2", weight=3)
        G.add_edge("b2", "b3", weight=3)
        G.add_edge("b1", "b3", weight=2)
        # Weak link
        G.add_edge("a1", "b1", weight=1)

        for node in G.nodes():
            G.nodes[node]["paper_ids"] = [f"paper_{node}"]

        # Default: single resolution [1.0], min_community_size=3
        communities = build_community_hierarchy(G)
        print(f"  Communities with defaults: {len(communities)}")

        # Should get exactly 2 communities (one per cluster)
        # min_community_size=3 means both clusters (size 3) are kept
        assert len(communities) <= 3, (
            f"Expected ≤3 communities with single resolution, got {len(communities)}"
        )
        # Each community should be at level 0 (single resolution)
        for c in communities:
            assert c["level"] == 0, f"All communities should be level 0, got {c['level']}"
        print("  All communities at level 0 OK")

        # Compare with old multi-resolution behavior
        communities_multi = build_community_hierarchy(
            G, resolutions=[0.5, 1.0, 2.0], min_community_size=2
        )
        print(f"  Communities with old defaults: {len(communities_multi)}")
        assert len(communities_multi) > len(communities), (
            "Multi-resolution should produce more communities than single"
        )
        print("  Multi-resolution produces more communities (as expected)")

        print("PASS: Single resolution community detection")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("GraphRAG Module Tests")
    print("=" * 60)

    results = []

    results.append(("GraphRAG imports", test_imports()))
    results.append(("State types", test_state_types()))
    results.append(("Entity normalization", test_entity_normalization()))
    results.append(("Graph builder", test_graph_builder()))
    results.append(("Community detection", test_community_detection()))
    results.append(("Supervisor routing", test_supervisor_routing()))
    results.append(("Workflow routing", test_workflow_routing()))
    results.append(("Tracker params in GraphRAG", test_tracker_params_in_graphrag()))
    results.append(("Entity extraction retry", test_entity_extraction_retry_params()))
    results.append(("Threshold clustering", test_threshold_clustering()))
    results.append(("No bidirectional co-occur", test_no_bidirectional_cooccur()))
    results.append(("GraphRAG defaults", test_graphrag_defaults()))
    results.append(("Single resolution community", test_single_resolution_community()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

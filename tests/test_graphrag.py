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
        assert state["token_tracker"] is None
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

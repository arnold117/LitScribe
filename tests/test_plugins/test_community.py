import pytest
import networkx as nx


def test_detect_empty_graph():
    from litscribe.plugins.graphrag.community_detector import detect_communities
    G = nx.Graph()
    result = detect_communities(G)
    assert result == []


def test_detect_small_graph_uses_connected_components():
    from litscribe.plugins.graphrag.community_detector import detect_communities
    G = nx.Graph()
    G.add_edge("a", "b", paper_ids=["p1"])
    G.add_edge("c", "d", paper_ids=["p2"])
    result = detect_communities(G)
    assert len(result) == 2  # Two connected components


def test_detect_larger_graph_uses_leiden():
    from litscribe.plugins.graphrag.community_detector import detect_communities
    G = nx.Graph()
    # Create two clusters
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(f"a{i}", f"a{j}", paper_ids=["p1"], weight=1)
    for i in range(5):
        for j in range(i + 1, 5):
            G.add_edge(f"b{i}", f"b{j}", paper_ids=["p2"], weight=1)
    # Weak bridge between clusters
    G.add_edge("a0", "b0", paper_ids=["p3"], weight=0.1)

    result = detect_communities(G)
    assert len(result) >= 1
    # All nodes should be assigned
    all_entities = []
    for c in result:
        all_entities.extend(c["entity_ids"])
    assert len(all_entities) == 10


def test_community_has_paper_ids():
    from litscribe.plugins.graphrag.community_detector import detect_communities
    G = nx.Graph()
    G.add_edge("x", "y", paper_ids=["p1", "p2"])
    result = detect_communities(G)
    assert len(result) == 1
    assert "p1" in result[0]["paper_ids"]

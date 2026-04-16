"""Tests for the GraphRAG optional plugin."""
import pytest
from unittest.mock import AsyncMock

from litscribe.models.analysis import PaperAnalysis


@pytest.mark.asyncio
async def test_graphrag_plugin_process():
    from litscribe.plugins.graphrag.plugin import GraphRAGPlugin

    analyses = [
        PaperAnalysis(
            paper_id="p1",
            key_findings=["Transformers use attention"],
            methodology="Experimental",
            relevance_score=0.9,
        )
    ]
    mock_llm = AsyncMock(
        return_value='[{"name": "Transformer", "entity_type": "method", "description": "Neural architecture", "aliases": []}]'
    )
    plugin = GraphRAGPlugin(llm_call=mock_llm)
    result = await plugin.process(analyses)

    assert result is not None
    assert "entities" in result
    assert "mentions" in result
    assert "graph" in result
    assert "communities" in result


def test_graphrag_plugin_init():
    from litscribe.plugins.graphrag.plugin import GraphRAGPlugin

    mock_llm = AsyncMock()
    plugin = GraphRAGPlugin(llm_call=mock_llm)
    assert plugin.llm_call is mock_llm


@pytest.mark.asyncio
async def test_graphrag_plugin_extracts_entities():
    from litscribe.plugins.graphrag.plugin import GraphRAGPlugin

    analyses = [
        PaperAnalysis(
            paper_id="p1",
            key_findings=["BERT uses masked language modelling"],
            relevance_score=0.8,
        )
    ]
    mock_llm = AsyncMock(
        return_value='[{"name": "BERT", "entity_type": "method", "description": "Bidirectional encoder", "aliases": ["bert"]}]'
    )
    plugin = GraphRAGPlugin(llm_call=mock_llm)
    result = await plugin.process(analyses)

    assert len(result["entities"]) == 1
    entity = list(result["entities"].values())[0]
    assert entity["name"] == "BERT"
    assert entity["entity_type"] == "method"


@pytest.mark.asyncio
async def test_graphrag_plugin_handles_invalid_json():
    """Plugin should tolerate LLM returning invalid JSON (skips that analysis)."""
    from litscribe.plugins.graphrag.plugin import GraphRAGPlugin

    analyses = [
        PaperAnalysis(paper_id="p1", key_findings=["something"], relevance_score=0.5)
    ]
    mock_llm = AsyncMock(return_value="not valid json at all")
    plugin = GraphRAGPlugin(llm_call=mock_llm)
    result = await plugin.process(analyses)

    # Should not raise; entities will be empty
    assert "entities" in result
    assert len(result["entities"]) == 0


@pytest.mark.asyncio
async def test_graphrag_co_occurrence_edges():
    """Two entities from the same paper should share an edge in the graph."""
    from litscribe.plugins.graphrag.plugin import GraphRAGPlugin

    analyses = [
        PaperAnalysis(
            paper_id="p1",
            key_findings=["Transformers and BERT both use attention"],
            relevance_score=0.9,
        )
    ]
    mock_llm = AsyncMock(
        return_value=(
            '[{"name": "Transformer", "entity_type": "method", "description": "", "aliases": []},'
            ' {"name": "BERT", "entity_type": "method", "description": "", "aliases": []}]'
        )
    )
    plugin = GraphRAGPlugin(llm_call=mock_llm)
    result = await plugin.process(analyses)

    graph = result["graph"]
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1


@pytest.mark.asyncio
async def test_linker_deduplicates():
    from litscribe.plugins.graphrag.linker import link_entities

    entities = [
        {"entity_id": "abc", "name": "BERT", "entity_type": "method", "aliases": ["bert"],
         "description": "d1", "paper_ids": ["p1"], "frequency": 1},
        {"entity_id": "abc", "name": "BERT", "entity_type": "method", "aliases": ["Bert"],
         "description": "d2", "paper_ids": ["p2"], "frequency": 1},
    ]
    linked = link_entities(entities)
    assert len(linked) == 1
    assert linked[0]["frequency"] == 2
    assert set(linked[0]["paper_ids"]) == {"p1", "p2"}


def test_detect_communities_single_component():
    from litscribe.plugins.graphrag.community_detector import detect_communities
    import networkx as nx

    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")

    communities = detect_communities(G)
    assert len(communities) == 1
    assert set(communities[0]["entity_ids"]) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_summarizer_fills_summaries():
    from litscribe.plugins.graphrag.summarizer import summarize_communities

    communities = [
        {
            "community_id": "0",
            "level": 0,
            "entity_ids": ["a", "b"],
            "paper_ids": ["p1"],
            "summary": "",
            "parent_id": None,
            "children_ids": [],
        }
    ]
    mock_llm = AsyncMock(return_value="This community studies attention mechanisms.")
    result = await summarize_communities(communities, mock_llm)

    assert len(result) == 1
    assert result[0]["summary"] == "This community studies attention mechanisms."
    mock_llm.assert_called_once()


@pytest.mark.asyncio
async def test_summarizer_skips_already_summarized():
    from litscribe.plugins.graphrag.summarizer import summarize_communities

    communities = [
        {
            "community_id": "0",
            "level": 0,
            "entity_ids": ["a"],
            "paper_ids": ["p1"],
            "summary": "Existing summary.",
            "parent_id": None,
            "children_ids": [],
        }
    ]
    mock_llm = AsyncMock()
    await summarize_communities(communities, mock_llm)

    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_summarizer_empty_input():
    from litscribe.plugins.graphrag.summarizer import summarize_communities

    result = await summarize_communities([], AsyncMock())
    assert result == []

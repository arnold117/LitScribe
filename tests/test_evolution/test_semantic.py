import pytest


@pytest.fixture
def semantic(tmp_data_dir):
    from litscribe.evolution.semantic import SemanticMemory
    from litscribe.store.vectors import VectorStore
    vectors = VectorStore(tmp_data_dir / "vectors")
    return SemanticMemory(vectors)


def test_absorb_and_search(semantic):
    from litscribe.models.analysis import PaperAnalysis
    analyses = [
        PaperAnalysis(paper_id="p1", key_findings=["Transformers enable parallel processing", "Attention is all you need"], methodology="Experimental", relevance_score=0.9),
        PaperAnalysis(paper_id="p2", key_findings=["BERT achieves SOTA on GLUE", "Pre-training helps downstream tasks"], methodology="Benchmark evaluation", relevance_score=0.8),
    ]
    semantic.absorb(analyses)
    results = semantic.search("transformer architecture")
    assert len(results) >= 1


def test_update_user_profile(semantic):
    semantic.update_user_profile(user_id="default", domain="NLP/AI", preferences={"tier": "standard", "language": "en"})
    profile = semantic.get_user_profile("default")
    assert profile is not None
    assert "NLP" in profile["document"]


def test_search_empty(semantic):
    results = semantic.search("nonexistent topic")
    assert results == []

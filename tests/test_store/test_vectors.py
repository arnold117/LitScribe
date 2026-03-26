import pytest


@pytest.fixture
def vector_store(tmp_data_dir):
    from litscribe.store.vectors import VectorStore
    store = VectorStore(tmp_data_dir / "vectors")
    return store


def test_add_and_search(vector_store):
    vector_store.add_texts(
        collection="test",
        texts=["LLM reasoning with chain of thought", "Protein folding prediction"],
        metadatas=[{"domain": "NLP"}, {"domain": "biology"}],
        ids=["doc1", "doc2"],
    )
    results = vector_store.search("reasoning language models", collection="test", n=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"


def test_search_empty_collection(vector_store):
    results = vector_store.search("anything", collection="empty", n=5)
    assert results == []


def test_add_to_named_collections(vector_store):
    vector_store.add_texts(
        collection="semantic_memory",
        texts=["Knowledge chunk 1"],
        metadatas=[{"source": "paper1"}],
        ids=["k1"],
    )
    vector_store.add_texts(
        collection="skill_embeddings",
        texts=["NLP search strategy"],
        metadatas=[{"skill": "nlp_search"}],
        ids=["s1"],
    )
    sem_results = vector_store.search("knowledge", collection="semantic_memory", n=1)
    skill_results = vector_store.search("search", collection="skill_embeddings", n=1)
    assert len(sem_results) == 1
    assert len(skill_results) == 1
    assert sem_results[0]["id"] == "k1"
    assert skill_results[0]["id"] == "s1"

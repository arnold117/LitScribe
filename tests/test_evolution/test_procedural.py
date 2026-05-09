import pytest


@pytest.fixture
def procedural(tmp_skills_dir, tmp_data_dir):
    from litscribe.evolution.procedural import ProceduralMemory
    from litscribe.store.vectors import VectorStore
    vectors = VectorStore(tmp_data_dir / "vectors")
    return ProceduralMemory(tmp_skills_dir, vectors)


def test_save_and_load_skill(procedural):
    procedural.save_skill(name="NLP Search Strategy", domain="NLP/AI", trigger="When research question involves NLP", strategy="1. Search arXiv cs.CL\n2. Snowball from top cited", learned_adjustments=[])
    skill = procedural.get_skill("nlp-search-strategy")
    assert skill is not None
    assert skill["name"] == "NLP Search Strategy"
    assert skill["domain"] == "NLP/AI"
    assert skill["version"] == 1


def test_list_skills(procedural):
    procedural.save_skill(name="Skill A", domain="NLP", trigger="t", strategy="s", learned_adjustments=[])
    procedural.save_skill(name="Skill B", domain="Bio", trigger="t", strategy="s", learned_adjustments=[])
    skills = procedural.list_skills()
    assert len(skills) == 2


def test_patch_skill_increments_version(procedural):
    procedural.save_skill(name="My Skill", domain="NLP", trigger="t", strategy="old strategy", learned_adjustments=[])
    procedural.patch_skill("my-skill", strategy="new strategy", adjustment="Added OpenAlex source")
    skill = procedural.get_skill("my-skill")
    assert skill["version"] == 2
    assert "new strategy" in skill["strategy"]
    assert "Added OpenAlex" in skill["raw_content"]


def test_find_relevant_skills(procedural):
    procedural.save_skill(name="NLP Search", domain="NLP", trigger="NLP queries", strategy="arxiv cs.CL", learned_adjustments=[])
    procedural.save_skill(name="Bio Search", domain="Biology", trigger="biomedical queries", strategy="pubmed mesh", learned_adjustments=[])
    results = procedural.find_relevant("NLP language models")
    assert len(results) >= 1
    assert results[0]["name"] == "NLP Search"


def test_get_nonexistent_skill(procedural):
    assert procedural.get_skill("nonexistent") is None

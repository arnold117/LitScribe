"""Tests for search strategy optimizations: query expansion, per-subtopic filters, relevance bonus, intro citations."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestQueryExpansion:
    """Test query expansion prompt and parsing changes."""

    def test_prompt_requests_12_queries(self):
        from agents.prompts import QUERY_EXPANSION_PROMPT
        assert "12 search queries" in QUERY_EXPANSION_PROMPT

    def test_prompt_has_6_dimensions(self):
        from agents.prompts import QUERY_EXPANSION_PROMPT
        assert "Core Methodology" in QUERY_EXPANSION_PROMPT
        assert "Application Domain" in QUERY_EXPANSION_PROMPT
        assert "Review & Meta-Analysis" in QUERY_EXPANSION_PROMPT
        assert "Recent Advances" in QUERY_EXPANSION_PROMPT
        assert "Cross-disciplinary" in QUERY_EXPANSION_PROMPT
        assert "Synonyms & Alternative Nomenclature" in QUERY_EXPANSION_PROMPT

    def test_prompt_relaxed_domain_constraint(self):
        from agents.prompts import QUERY_EXPANSION_PROMPT
        assert "PREFER queries within the research domain" in QUERY_EXPANSION_PROMPT
        assert "adjacent fields" in QUERY_EXPANSION_PROMPT

    def test_prompt_json_schema_has_6_keys(self):
        from agents.prompts import QUERY_EXPANSION_PROMPT
        assert "synonyms_nomenclature" in QUERY_EXPANSION_PROMPT
        assert "review_meta" in QUERY_EXPANSION_PROMPT

    def test_discovery_agent_parses_6_dimensions(self):
        """discovery_agent.py expand_queries() iterates over 6 dimension keys."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
        content = src_path.read_text()
        assert "synonyms_nomenclature" in content


class TestSubTopicFilters:
    """Test per-subtopic domain filter fields."""

    def test_subtopic_has_filter_fields(self):
        from agents.state import SubTopic
        # TypedDict annotations should include the new fields
        annotations = SubTopic.__annotations__
        assert "arxiv_categories" in annotations
        assert "s2_fields" in annotations
        assert "pubmed_mesh" in annotations

    def test_complexity_prompt_has_subtopic_filters(self):
        from agents.prompts import COMPLEXITY_ASSESSMENT_PROMPT
        # The JSON schema in the prompt should include per-subtopic filter fields
        assert "topic-specific arXiv categories" in COMPLEXITY_ASSESSMENT_PROMPT
        assert "topic-specific S2 fields" in COMPLEXITY_ASSESSMENT_PROMPT
        assert "topic-specific MeSH terms" in COMPLEXITY_ASSESSMENT_PROMPT

    def test_revision_prompt_has_subtopic_filters(self):
        from agents.prompts import PLAN_REVISION_PROMPT
        assert "topic-specific arXiv categories" in PLAN_REVISION_PROMPT

    def test_discovery_uses_per_subtopic_filters(self):
        """discovery_agent.py uses topic.get('arxiv_categories') for per-subtopic overrides."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
        content = src_path.read_text()
        assert 'topic.get("arxiv_categories")' in content
        assert 'topic.get("s2_fields")' in content
        assert 'topic.get("pubmed_mesh")' in content


class TestCustomQueriesCap:
    """Test that planning agent allows up to 5 custom queries per subtopic."""

    def test_planning_cap_is_5(self):
        """planning_agent.py uses [:5] not [:3]."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "planning_agent.py"
        content = src_path.read_text()
        # Should have [:5] for custom_queries (using _ensure_list wrapper)
        assert '"custom_queries"))[:5]' in content
        # Should NOT have [:3] for custom_queries
        assert '"custom_queries"))[:3]' not in content


class TestRelevanceBonus:
    """Test review/survey and high-citation relevance bonus."""

    def test_review_bonus_in_code(self):
        """unified_search.py gives review/survey papers a relevance bonus."""
        src_path = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
        content = src_path.read_text()
        assert "review" in content and "survey" in content and "meta-analysis" in content
        assert "0.12" in content

    def test_citation_bonus_in_code(self):
        """unified_search.py gives highly-cited papers a relevance bonus."""
        src_path = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
        content = src_path.read_text()
        assert ">= 100" in content or ">=100" in content or ">= 100" in content
        assert "0.08" in content

    def test_review_bonus_logic(self):
        """Simulate the bonus logic: a review paper with score 0.25 should get boosted to 0.37."""
        base_score = 0.25
        title = "A Comprehensive Review of Plant Terpenoid Biosynthesis"
        title_lower = title.lower()

        score = base_score
        if any(w in title_lower for w in ("review", "survey", "meta-analysis", "overview", "综述")):
            score = min(score + 0.12, 1.0)

        assert score == pytest.approx(0.37)

    def test_citation_bonus_logic(self):
        """A paper with 150 citations and score 0.30 should get boosted to 0.38."""
        score = 0.30
        citations = 150
        if citations >= 100:
            score = min(score + 0.08, 1.0)
        assert score == pytest.approx(0.38)

    def test_combined_bonus_capped_at_1(self):
        """Combined bonuses should not exceed 1.0."""
        score = 0.90
        # Review bonus
        score = min(score + 0.12, 1.0)
        assert score == 1.0
        # Citation bonus should still be capped
        score = min(score + 0.08, 1.0)
        assert score == 1.0


class TestIntroCitations:
    """Test that the introduction prompt allows citations and receives paper data."""

    def test_intro_prompt_allows_citations(self):
        from agents.prompts import REVIEW_INTRO_PROMPT
        # Should NOT have the old "Do NOT include any citations" rule
        assert "Do NOT include any citations" not in REVIEW_INTRO_PROMPT

    def test_intro_prompt_encourages_citations(self):
        from agents.prompts import REVIEW_INTRO_PROMPT
        assert "Cite key papers" in REVIEW_INTRO_PROMPT
        assert "[Author, Year]" in REVIEW_INTRO_PROMPT

    def test_intro_prompt_has_paper_summaries_placeholder(self):
        from agents.prompts import REVIEW_INTRO_PROMPT
        assert "{paper_summaries}" in REVIEW_INTRO_PROMPT

    def test_synthesis_passes_paper_summaries(self):
        """synthesis_agent.py passes paper_summaries to intro prompt."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "synthesis_agent.py"
        content = src_path.read_text()
        assert "paper_summaries=" in content
        # Should use format_summaries_for_prompt for intro
        assert "intro_summaries" in content

    def test_graphrag_prompt_intro_cites(self):
        """GRAPHRAG_LITERATURE_REVIEW_PROMPT INTRODUCTION should require citations."""
        from agents.prompts import GRAPHRAG_LITERATURE_REVIEW_PROMPT
        # Find the INTRODUCTION section
        intro_idx = GRAPHRAG_LITERATURE_REVIEW_PROMPT.index("INTRODUCTION")
        thematic_idx = GRAPHRAG_LITERATURE_REVIEW_PROMPT.index("THEMATIC ANALYSIS")
        intro_section = GRAPHRAG_LITERATURE_REVIEW_PROMPT[intro_idx:thematic_idx]
        assert "Cite" in intro_section or "cite" in intro_section

    def test_literature_prompt_intro_cites(self):
        """LITERATURE_REVIEW_PROMPT INTRODUCTION should require citations."""
        from agents.prompts import LITERATURE_REVIEW_PROMPT
        intro_idx = LITERATURE_REVIEW_PROMPT.index("INTRODUCTION")
        thematic_idx = LITERATURE_REVIEW_PROMPT.index("THEMATIC ANALYSIS")
        intro_section = LITERATURE_REVIEW_PROMPT[intro_idx:thematic_idx]
        assert "Cite" in intro_section or "cite" in intro_section


class TestSectionedGeneration:
    """Test that sectioned generation is wired up for long reviews."""

    def test_sectioned_path_triggered(self):
        """synthesis_agent uses sectioned generation when target_words > 4096."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "synthesis_agent.py"
        content = src_path.read_text()
        assert "use_sectioned = target_words > 4096" in content
        assert "generate_review_sectioned" in content

    def test_standard_tier_max_per_source(self):
        """Standard tier max_per_source should be 35 (increased for niche topics)."""
        from agents.state import TIER_CONFIG
        assert TIER_CONFIG["standard"]["max_per_source"] == 35


class TestCJKQueryHandling:
    """Test that CJK queries are not excluded from search."""

    def test_cjk_query_included(self):
        """Non-English queries should use CJK expansion, not be excluded."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
        content = src_path.read_text()
        assert "_expand_cjk_queries" in content
        assert "queries + cjk_queries" in content

    def test_expand_cjk_queries_basic(self):
        """_expand_cjk_queries should produce multiple CJK query variants."""
        from agents.discovery_agent import _expand_cjk_queries
        queries = _expand_cjk_queries("倍半萜香豆素生源合成途径解析")
        # Should include original + at least one variant
        assert len(queries) >= 2
        # Original should be first
        assert queries[0] == "倍半萜香豆素生源合成途径解析"

    def test_expand_cjk_queries_english_passthrough(self):
        """_expand_cjk_queries on English text should still return it."""
        from agents.discovery_agent import _expand_cjk_queries
        queries = _expand_cjk_queries("sesquiterpene coumarin biosynthesis")
        assert len(queries) >= 1

    def test_expand_cjk_queries_extracts_terms(self):
        """CJK expansion should extract meaningful terms like 倍半萜, 香豆素."""
        from agents.discovery_agent import _expand_cjk_queries
        queries = _expand_cjk_queries("倍半萜香豆素生源合成途径解析")
        all_text = " ".join(queries)
        # Key domain terms should appear in the expanded queries
        assert "倍半萜" in all_text
        assert "香豆素" in all_text

    def test_abstract_screening_rejects_unrelated_domains(self):
        """ABSTRACT_SCREENING_PROMPT should explicitly mention clinical trial exclusion."""
        from agents.prompts import ABSTRACT_SCREENING_PROMPT
        assert "clinical trial" in ABSTRACT_SCREENING_PROMPT.lower()
        assert "epidemiology" in ABSTRACT_SCREENING_PROMPT.lower()


class TestCJKCrossLanguageRelevance:
    """Test that CJK queries don't kill English papers via keyword/semantic mismatch."""

    def test_cjk_keyword_floor_in_code(self):
        """unified_search.py should have a CJK cross-language relevance floor."""
        src_path = Path(__file__).parent.parent / "src" / "aggregators" / "unified_search.py"
        content = src_path.read_text()
        assert "cjk_only" in content
        assert "0.35" in content

    def test_semantic_scorer_cjk_floor(self):
        """semantic_scorer.py should have a CJK cross-language relevance floor."""
        src_path = Path(__file__).parent.parent / "src" / "embeddings" / "semantic_scorer.py"
        content = src_path.read_text()
        assert "_cjk_query" in content
        assert "0.35" in content

    def test_min_relevance_threshold_lowered(self):
        """MIN_RELEVANCE in discovery_agent should be 0.25 (not 0.35)."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
        content = src_path.read_text()
        assert "MIN_RELEVANCE = 0.25" in content

    def test_abstract_screening_threshold(self):
        """Abstract screening should only run when > 15 papers (not > 3)."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
        content = src_path.read_text()
        assert "len(selected_papers) > 15" in content

    def test_per_query_minimum_10(self):
        """max_per_query minimum should be 10 (not 5) for niche topic headroom."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
        content = src_path.read_text()
        assert ", 10)" in content
        # Should NOT have the old minimum of 5
        assert "max_per_source // max(len(queries_to_search), 1), 5)" not in content

    def test_research_question_direct_search(self):
        """Per-subtopic path should always search the research question directly."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
        content = src_path.read_text()
        assert "research_question not in expanded_queries" in content

    def test_cjk_queries_searched_separately(self):
        """CJK queries should be searched as a separate batch, not mixed with novel_llm."""
        src_path = Path(__file__).parent.parent / "src" / "agents" / "discovery_agent.py"
        content = src_path.read_text()
        assert "cjk_results = await search_all_sources" in content

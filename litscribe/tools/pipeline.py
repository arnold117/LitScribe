from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI

from litscribe.config import Config
from litscribe.models.paper import Paper
from litscribe.models.plan import ResearchPlan, SubTopic
from litscribe.models.review import ReviewOutput
from litscribe.tools.status import PipelineState

logger = logging.getLogger(__name__)


async def _call_llm(model: ChatOpenAI, prompt: str) -> str:
    result = await model.ainvoke(prompt)
    return result.content


async def _call_llm_json(model: ChatOpenAI, prompt: str, retries: int = 2) -> dict | list:
    import re
    for attempt in range(retries + 1):
        raw = await _call_llm(model, prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == retries:
                logger.warning(f"JSON parse failed after {retries+1} attempts")
                return {}
    return {}


FAST_PLAN_PROMPT = """Decompose this research question into 2-4 sub-topics for a literature review.

Research Question: {research_question}

Output JSON:
{{
  "domain": "Primary academic field (e.g. Biology, Computer Science, Chemistry)",
  "sub_topics": [
    {{"name": "sub-topic name", "custom_queries": ["search query 1", "search query 2"]}}
  ]
}}

Keep queries in English, 3-8 words each, optimized for academic search engines."""


async def step_plan(model: ChatOpenAI, state: PipelineState) -> None:
    logger.info("Pipeline step: PLAN")
    t = time.time()

    prompt = FAST_PLAN_PROMPT.format(research_question=state.research_question)
    result = await _call_llm_json(model, prompt)

    if isinstance(result, dict) and result.get("sub_topics"):
        sub_topics = []
        for st in result["sub_topics"]:
            sub_topics.append(SubTopic(
                name=st.get("name", ""),
                keywords=st.get("custom_queries", st.get("keywords", [])),
                estimated_papers=st.get("estimated_papers", 10),
            ))
        state.plan = ResearchPlan(
            question=state.research_question,
            sub_topics=sub_topics,
            domain=result.get("domain", "General"),
            tier="standard",
            max_papers=40,
            language=state.language,
            target_words=max(1000, sum(st.estimated_papers for st in sub_topics) * 130),
        )
        state.domain = result.get("domain", "General")
    else:
        state.plan = ResearchPlan(
            question=state.research_question,
            sub_topics=[SubTopic(name=state.research_question, keywords=[state.research_question], estimated_papers=20)],
            domain="General", tier="standard", max_papers=40,
            language=state.language, target_words=3000,
        )

    logger.info(f"  PLAN done: {len(state.plan.sub_topics)} sub-topics, domain={state.domain} ({time.time()-t:.1f}s)")


async def step_search(model: ChatOpenAI, state: PipelineState, config: Config, max_papers: int = 40) -> None:
    from litscribe.tools.search import search_all_sources

    logger.info("Pipeline step: SEARCH")
    t = time.time()

    queries = [state.research_question]
    if state.plan:
        for st in state.plan.sub_topics:
            queries.extend(st.keywords[:2])
    if state.extra_queries:
        queries.extend(state.extra_queries)

    # Cross-lingual: if question is non-English, keep original + English queries
    # If question is English, check if domain benefits from CJK queries
    has_cjk = any('一' <= c <= '鿿' for c in state.research_question)
    if has_cjk:
        queries.insert(0, state.research_question)  # ensure original CJK is first

    seen: set[str] = set()
    unique = []
    for q in queries:
        ql = q.strip().lower()
        if ql and ql not in seen:
            seen.add(ql)
            unique.append(q.strip())

    papers = await search_all_sources(unique[:6], config, max_per_source=max_papers, domain=state.domain)

    # Coarse keyword prefilter (English terms only; CJK whole-phrases never
    # substring-match English abstracts). Terms are IDF-weighted so common
    # words (power, time, model) don't let off-topic papers rank high on
    # coincidental matches — the real ranking is the LLM selection below.
    _STOP = {"with", "from", "that", "this", "have", "been", "their", "using",
             "based", "study", "analysis", "approach", "method", "novel",
             "recent", "review", "paper", "results"}
    core_terms = {
        w for q in unique[:3] for w in q.lower().split()
        if len(w) >= 4 and w not in _STOP and not any("一" <= c <= "鿿" for c in w)
    }

    if core_terms:
        import math
        texts = [f"{p.title} {p.abstract or ''}".lower() for p in papers]
        df = {t: sum(1 for txt in texts if t in txt) or 1 for t in core_terms}
        n = len(papers) or 1
        weight = {t: math.log(1 + n / df[t]) for t in core_terms}

        def _relevance(p):
            text = f"{p.title} {p.abstract or ''}".lower()
            return sum(weight[t] for t in core_terms if t in text)

        papers.sort(key=_relevance, reverse=True)
        before = len(papers)
        matched = [p for p in papers if _relevance(p) > 0]
        # keep a generous candidate pool for the LLM; only fall back to raw
        # order if almost nothing matched
        papers = matched if len(matched) >= 5 else papers
        if len(papers) != before:
            logger.info(f"  Keyword prefilter: {before} → {len(papers)} (weighted terms {list(core_terms)[:5]})")

    # LLM-based paper selection is the primary relevance gate: whenever there
    # are more candidates than we need, let the LLM pick the most relevant
    # (the keyword prefilter only coarsely ranked/trimmed the pool).
    if len(papers) > max_papers:
        from litscribe.prompts.utils import format_papers_for_prompt
        papers_text = format_papers_for_prompt([p.model_dump() for p in papers[:30]], max_chars=10000)
        select_prompt = (
            f"Select the {max_papers} most relevant papers for a literature review on: {state.research_question}\n\n"
            f"Papers:\n{papers_text}\n\n"
            f"Return ONLY a JSON array of paper IDs (the [ID] values): [\"id1\", \"id2\", ...]"
        )
        try:
            result = await _call_llm_json(model, select_prompt)
            if isinstance(result, list) and result:
                id_set = set(str(s) for s in result)
                selected = [p for p in papers if p.paper_id in id_set]
                if len(selected) >= 3:
                    papers = selected
                    logger.info(f"  LLM selection: {len(papers)} papers chosen")
        except Exception as e:
            logger.debug(f"  LLM selection failed: {e}")

    state.papers = papers[:max_papers]
    state.iteration += 1

    # Enrich PDF URLs via Unpaywall (for full-text analysis)
    try:
        from litscribe.services.unpaywall import enrich_pdf_urls
        email = getattr(getattr(config, "services", None), "ncbi_email", "") or ""
        n = await enrich_pdf_urls(state.papers, email=email)
        if n:
            logger.info(f"  Unpaywall: {n} papers enriched with OA PDF URLs")
    except Exception as e:
        logger.debug(f"  Unpaywall enrichment skipped: {e}")

    logger.info(f"  SEARCH done: {len(papers)} found → {len(state.papers)} kept ({time.time()-t:.1f}s)")


SAFE_PDF_DOMAINS = {
    "arxiv.org", "ncbi.nlm.nih.gov", "pmc", "doi.org",
    "nature.com", "wiley.com", "springer.com", "elsevier.com",
    "sciencedirect.com", "plos.org", "mdpi.com", "biomedcentral.com",
    "biorxiv.org", "medrxiv.org", "frontiersin.org", "acs.org",
    "rsc.org", "ieee.org", "acm.org",
}


def _is_safe_pdf_url(url: str) -> bool:
    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""
    return any(domain in host for domain in SAFE_PDF_DOMAINS)


async def _try_get_full_text(paper: Paper) -> str | None:
    if not paper.pdf_urls:
        return None

    url = paper.pdf_urls[0]
    if not _is_safe_pdf_url(url):
        return None

    try:
        from litscribe.services.pdf import PDFService
        pdf_svc = PDFService()
        parsed = await pdf_svc.parse(url)
        if parsed and parsed.markdown and len(parsed.markdown) > 200:
            return parsed.markdown[:8000]
    except Exception as _e:
        logger.debug(f"Silent error: {_e}")
    return None


async def step_read(model: ChatOpenAI, state: PipelineState) -> None:
    import asyncio
    from litscribe.prompts.reading import ABSTRACT_ONLY_ANALYSIS_PROMPT, COMBINED_PAPER_ANALYSIS_PROMPT
    from litscribe.models.analysis import PaperAnalysis

    logger.info(f"Pipeline step: READ ({len(state.papers)} papers)")
    t = time.time()

    async def analyze_one(paper: Paper) -> PaperAnalysis:
        full_text = await _try_get_full_text(paper)

        if full_text:
            prompt = COMBINED_PAPER_ANALYSIS_PROMPT.format(
                research_question=state.research_question,
                title=paper.title,
                authors=", ".join(paper.authors[:3]) if paper.authors else "Unknown",
                year=paper.year or "N/A",
                abstract=paper.abstract or "(no abstract)",
                full_text=full_text,
            )
        else:
            prompt = ABSTRACT_ONLY_ANALYSIS_PROMPT.format(
                research_question=state.research_question,
                title=paper.title,
                authors=", ".join(paper.authors[:3]) if paper.authors else "Unknown",
                year=paper.year or "N/A",
                venue=paper.venue or "",
                abstract=paper.abstract or "(no abstract)",
                metadata_section=f"Citations: {paper.citations or 0}",
            )
        try:
            result = await _call_llm_json(model, prompt, retries=1)
            if isinstance(result, dict):
                return PaperAnalysis(
                    paper_id=paper.paper_id,
                    key_findings=result.get("key_findings", []),
                    methodology=result.get("methodology", ""),
                    strengths=result.get("strengths", []),
                    limitations=result.get("limitations", []),
                    relevance_score=float(result.get("relevance_to_question", 0.5)),
                    themes=[],
                )
        except Exception as e:
            logger.warning(f"Analysis failed for {paper.paper_id}: {e}")

        return PaperAnalysis(
            paper_id=paper.paper_id,
            key_findings=[f"Abstract: {(paper.abstract or '')[:200]}"],
            methodology="Analysis unavailable", strengths=[], limitations=[],
            relevance_score=0.5, themes=[],
        )

    batch_size = 5
    analyses = []
    for i in range(0, len(state.papers), batch_size):
        batch = state.papers[i:i + batch_size]
        results = await asyncio.gather(*[analyze_one(p) for p in batch], return_exceptions=True)
        for r in results:
            if isinstance(r, PaperAnalysis):
                analyses.append(r)

    # Adaptive relevance filter — stricter when many papers, lenient when few
    if len(analyses) > 8:
        threshold = 0.5
    elif len(analyses) > 4:
        threshold = 0.3
    else:
        threshold = 0.2
    relevant = [a for a in analyses if a.relevance_score >= threshold]
    dropped = len(analyses) - len(relevant)
    if dropped:
        logger.info(f"  Relevance filter: dropped {dropped} papers (score < 0.3)")
        # Also remove from state.papers
        relevant_ids = {a.paper_id for a in relevant}
        state.papers = [p for p in state.papers if p.paper_id in relevant_ids]

    state.analyses = relevant
    logger.info(f"  READ done: {len(relevant)} relevant / {len(analyses)} analyzed ({time.time()-t:.1f}s)")


async def step_contradictions(model: ChatOpenAI, state: PipelineState) -> None:
    if len(state.analyses) < 2:
        return

    from litscribe.tools.contradictions import detect_contradictions

    logger.info(f"Pipeline step: CONTRADICTIONS ({len(state.analyses)} papers)")
    t = time.time()

    report = await detect_contradictions(model, state.analyses)
    state.contradiction_report = report
    logger.info(f"  CONTRADICTIONS done: {report.count} found ({time.time()-t:.1f}s)")


async def step_graphrag(model: ChatOpenAI, state: PipelineState) -> None:
    if len(state.analyses) < 5:
        logger.info(f"Pipeline step: GRAPHRAG skipped ({len(state.analyses)} < 5)")
        return

    from litscribe.tools.graphrag import build_knowledge_graph

    logger.info("Pipeline step: GRAPHRAG")
    t = time.time()

    async def llm_call(prompt: str, **kwargs) -> str:
        return await _call_llm(model, prompt)

    try:
        state.graph = await build_knowledge_graph(state.analyses, llm_call)
    except Exception as e:
        logger.warning(f"GraphRAG failed: {e}, skipping")
        state.graph = None
    n = len(state.graph.get("communities", [])) if state.graph else 0
    logger.info(f"  GRAPHRAG done: {n} communities ({time.time()-t:.1f}s)")


async def step_synthesize(model: ChatOpenAI, state: PipelineState, user_instructions: str = "") -> None:
    from litscribe.tools.synthesis import synthesize

    logger.info("Pipeline step: SYNTHESIZE")
    t = time.time()

    review = await synthesize(
        router=None, analyses=state.analyses,
        research_question=state.research_question,
        language=state.language, graph_context=state.graph,
        user_instructions=user_instructions, papers=state.papers,
        model=model,
    )
    # Append reference list with citation keys
    from litscribe.tools.cite_keys import assign_cite_keys
    key_map = assign_cite_keys(state.papers)

    ref_lines = ["\n\n## References\n"]
    for p in state.papers:
        key = key_map.get(p.paper_id, "unknown")
        authors = ", ".join(p.authors[:3]) if p.authors else "Unknown"
        if len(p.authors) > 3:
            authors += " et al."
        venue = f" *{p.venue}*." if p.venue else ""
        doi = f" doi:{p.doi}" if p.doi else ""
        ref_lines.append(f"[{key}]: {authors} ({p.year}). {p.title}.{venue}{doi}")

    review_with_refs = ReviewOutput(
        text=review.text + "\n".join(ref_lines),
        citations=review.citations,
        themes=review.themes,
        word_count=review.word_count,
        language=review.language,
    )
    # Generate appendix sections in parallel
    try:
        import asyncio as _aio
        from litscribe.tools.comparison import generate_comparison_table, generate_timeline
        from litscribe.tools.analytics import extract_statistics, stats_to_markdown_table, suggest_figures

        themes_list = [t.name for t in review.themes] if review.themes else []

        comp_task = generate_comparison_table(model, state.papers, state.analyses, key_map)
        timeline_task = generate_timeline(model, state.papers, state.analyses, key_map)
        stats_task = extract_statistics(model, state.papers, state.analyses, key_map)
        figures_task = suggest_figures(model, review.text[:2000], themes_list, len(state.papers))

        comp_table, timeline, stats, figures = await _aio.gather(
            comp_task, timeline_task, stats_task, figures_task,
            return_exceptions=True,
        )

        appendix = ""
        if isinstance(comp_table, str) and comp_table:
            appendix += f"\n\n## Methodology Comparison\n\n{comp_table}"
        if isinstance(timeline, str) and timeline:
            appendix += f"\n\n## Research Timeline\n\n{timeline}"
        if isinstance(stats, list) and stats:
            appendix += f"\n\n## Statistical Summary\n\n{stats_to_markdown_table(stats)}"
        figures = figures if isinstance(figures, list) else []
        if figures:
            appendix += "\n\n## Suggested Figures\n"
            for f in figures:
                appendix += f"\n- **{f.get('title', 'Figure')}** ({f.get('type', '')}): {f.get('description', '')} — {f.get('placement', '')}"

        if appendix:
            review_with_refs = ReviewOutput(
                text=review_with_refs.text + appendix,
                citations=review_with_refs.citations,
                themes=review_with_refs.themes,
                word_count=review_with_refs.word_count,
                language=review_with_refs.language,
            )
    except Exception as e:
        logger.debug(f"Comparison/timeline/stats generation failed: {e}")

    state.synthesis = review_with_refs
    logger.info(f"  SYNTHESIZE done: {review.word_count} words, {len(review.themes)} themes ({time.time()-t:.1f}s)")


async def step_debate(model: ChatOpenAI, state: PipelineState) -> None:
    if state.synthesis is None:
        return

    from litscribe.tools.debate import multi_round_debate

    logger.info("Pipeline step: DEBATE (reviewer ↔ synthesizer)")
    t = time.time()

    revised, critiques = await multi_round_debate(
        model, state.synthesis, state.research_question,
        state.analyses, state.papers, max_rounds=2,
    )
    state.synthesis = revised
    total_issues = sum(len(c.get("issues", [])) for c in critiques)
    logger.info(f"  DEBATE done: {len(critiques)} rounds, {total_issues} issues addressed ({time.time()-t:.1f}s)")


async def step_ground(model: ChatOpenAI, state: PipelineState) -> None:
    from litscribe.tools.grounding import ground_citations, apply_fixes

    if state.synthesis is None:
        return

    logger.info("Pipeline step: GROUND (citation verification)")
    t = time.time()

    report = await ground_citations(
        model, state.synthesis.text, state.papers, state.analyses,
    )
    state.grounding_report = report

    if report.unsupported > 0:
        # Try auto-fix first
        fixed_text = apply_fixes(state.synthesis.text, report)

        # Then LLM fix pass: remove unsupported claims entirely
        unsupported_keys = [c.author for c in report.claims if c.supported is False]
        if unsupported_keys:
            fix_prompt = (
                f"Remove or rewrite sentences that cite these papers incorrectly: {unsupported_keys}\n\n"
                f"Current review:\n{fixed_text[:3000]}\n\n"
                f"Only remove/rewrite the specific unsupported claims. Keep everything else unchanged."
            )
            try:
                fix_result = await model.ainvoke(fix_prompt)
                fixed_text = fix_result.content.strip()
            except Exception as _e:
                logger.debug(f"Silent error: {_e}")

        if fixed_text != state.synthesis.text:
            state.synthesis = ReviewOutput(
                text=fixed_text,
                citations=state.synthesis.citations,
                themes=state.synthesis.themes,
                word_count=state.synthesis.word_count,
                language=state.synthesis.language,
            )
            logger.info(f"  GROUND: fixed {report.unsupported} unsupported claims")

    logger.info(
        f"  GROUND done: {report.verified}/{report.total_citations} verified, "
        f"{report.unsupported} fixed, accuracy={report.accuracy:.0%} ({time.time()-t:.1f}s)"
    )


async def step_review(model: ChatOpenAI, state: PipelineState) -> None:
    from litscribe.tools.review import evaluate_review

    logger.info("Pipeline step: REVIEW")
    t = time.time()

    assessment = await evaluate_review(
        router=None, review=state.synthesis, analyses=state.analyses,
        plan=state.plan, research_question=state.research_question,
        model=model,
    )
    state.assessment = assessment
    logger.info(f"  REVIEW done: score={assessment.score:.2f}, passed={assessment.passed} ({time.time()-t:.1f}s)")


async def run_review(
    model: ChatOpenAI,
    config: Config,
    state: PipelineState,
    max_papers: int = 40,
    user_instructions: str = "",
    memory=None,
) -> str:
    total_start = time.time()

    # Sanitize input
    from litscribe.tools.sanitize import sanitize_research_question
    state.research_question = sanitize_research_question(state.research_question)
    if user_instructions:
        from litscribe.tools.sanitize import sanitize_input
        user_instructions = sanitize_input(user_instructions, max_length=500)

    # 1. Plan
    await step_plan(model, state)

    # Inject prior knowledge from past reviews
    try:
        from litscribe.store.knowledge import KnowledgeStore
        kb = KnowledgeStore(config.db_path)
        prior = await kb.get_context_for_review(state.research_question, state.domain)
        if prior:
            user_instructions = f"{prior}\n\n{user_instructions}" if user_instructions else prior
            logger.info(f"  Injected prior knowledge ({len(prior)} chars)")
    except Exception as _e:
        logger.debug(f"Silent error: {_e}")

    for iteration in range(state.max_iterations):
        # 2. Search
        await step_search(model, state, config, max_papers)

        if len(state.papers) < 3:
            return f"Only found {len(state.papers)} papers. Try a broader research question."

        # 3. Read
        await step_read(model, state)

        # 4. Contradiction detection
        await step_contradictions(model, state)

        # 5. GraphRAG
        await step_graphrag(model, state)

        # 6. Synthesize (inject contradictions if found)
        contra_instructions = ""
        if state.contradiction_report and state.contradiction_report.count > 0:
            from litscribe.tools.contradictions import format_contradictions_for_synthesis
            from litscribe.tools.cite_keys import assign_cite_keys
            key_map = assign_cite_keys(state.papers)
            contra_text = format_contradictions_for_synthesis(state.contradiction_report, key_map)
            contra_instructions = (
                f"\n\nIMPORTANT: The following contradictions were detected between papers. "
                f"Include a 'Critical Analysis' or 'Contradictions' section discussing these:\n{contra_text}"
            )
        full_instructions = (user_instructions + contra_instructions).strip()
        await step_synthesize(model, state, full_instructions)

        # 7. Debate (reviewer ↔ synthesizer)
        await step_debate(model, state)

        # 8. Citation grounding
        await step_ground(model, state)

        # 9. Claim-level contradiction check (post-synthesis)
        if state.synthesis:
            try:
                from litscribe.tools.contradictions import detect_claim_contradictions
                claim_contras = await detect_claim_contradictions(model, state.synthesis.text, state.analyses)
                if claim_contras:
                    logger.info(f"  Claim-level contradictions: {len(claim_contras)} found")
            except Exception as _e:
                logger.debug(f"Silent error: {_e}")

        # 10. Review
        await step_review(model, state)

        # 11. Metacognitive evaluation
        if state.assessment and state.assessment.score < 0.8:
            try:
                from litscribe.tools.metacognition import metacognitive_evaluate, save_strategy
                meta = await metacognitive_evaluate(model, state)
                if meta.get("strategy_adjustment"):
                    await save_strategy(config, state.domain, meta["strategy_adjustment"])

                if meta.get("should_rerun") and iteration < state.max_iterations - 1:
                    steps = meta.get("steps_to_rerun", [])
                    logger.info(f"Metacognition: re-running {steps}")
                    if "SEARCH" in steps:
                        state.extra_queries = getattr(state.assessment, "refined_queries", []) or []
                        state.synthesis = None
                        state.assessment = None
                        state.graph = None
                        continue
                    if "SYNTHESIZE" in steps:
                        state.synthesis = None
                        state.assessment = None
                        continue
            except Exception as e:
                logger.debug(f"Metacognition failed: {e}")

        # Check if good enough
        if state.assessment.passed or state.assessment.score >= 0.65:
            break

        # Standard loop-back
        if iteration < state.max_iterations - 1:
            logger.info(f"Loop-back: score={state.assessment.score:.2f}, refining queries")
            state.extra_queries = getattr(state.assessment, "refined_queries", []) or []
            state.synthesis = None
            state.assessment = None
            state.graph = None

    # Evolution: post-task evaluate
    if memory and state.assessment:
        try:
            from litscribe.evolution.skill_evolver import TaskMetrics
            metrics = TaskMetrics(
                sub_topic_count=len(state.plan.sub_topics) if state.plan else 0,
                papers_found=len(state.papers),
                papers_relevant=len(state.analyses),
                loop_back_count=max(0, state.iteration - 1),
                source_count=len({s for p in state.papers for s in p.sources}),
            )
            themes = [t.name for t in state.synthesis.themes] if state.synthesis else []
            await memory.evolver.post_task_evaluate(
                session_id=f"review-{int(total_start)}",
                question=state.research_question,
                score=state.assessment.score,
                metrics=metrics,
                domain=state.domain,
                trace_summary=f"Pipeline: {len(state.papers)} papers, {len(state.analyses)} analyzed, themes: {themes}",
            )
            logger.info(f"Evolution: skill evaluated (score={state.assessment.score:.2f})")
        except Exception as e:
            logger.warning(f"Evolution post_task_evaluate failed: {e}")

    # Save knowledge for future reviews
    if state.analyses:
        try:
            from litscribe.store.knowledge import KnowledgeStore
            from litscribe.tools.cite_keys import assign_cite_keys
            kb = KnowledgeStore(config.db_path)
            km = assign_cite_keys(state.papers) if state.papers else {}
            await kb.save_findings(state.domain, state.research_question, state.analyses, km)
        except Exception as e:
            logger.debug(f"Knowledge save failed: {e}")

    # Save session
    session_id = ""
    try:
        from litscribe.store.sessions import SessionStore
        store = SessionStore(config.db_path)
        session_id = await store.save_session(state)
        logger.info(f"Session saved: {session_id}")
    except Exception as e:
        logger.warning(f"Session save failed: {e}")

    elapsed = time.time() - total_start
    logger.info(f"Pipeline complete in {elapsed:.1f}s")

    return (
        f"Literature review complete (session: {session_id}).\n"
        f"- Papers: {len(state.papers)} found, {len(state.analyses)} analyzed\n"
        f"- Review: {state.synthesis.word_count} words, {len(state.synthesis.themes)} themes\n"
        f"- Score: {state.assessment.score:.2f}\n"
        f"- Time: {elapsed:.0f}s\n\n"
        f"{state.synthesis.text[:2000]}"
    )

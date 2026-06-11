from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from litscribe.config import Config
from litscribe.agents import _build_model
from litscribe.tools.status import PipelineState
from litscribe.models.review import ReviewOutput

logger = logging.getLogger(__name__)

app = FastAPI(title="LitScribe", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Simple rate limiter
_request_times: dict[str, list[float]] = {}
RATE_LIMIT = 10  # requests per minute per endpoint

async def check_rate_limit(endpoint: str):
    import time as _time
    now = _time.time()
    times = _request_times.setdefault(endpoint, [])
    times[:] = [t for t in times if now - t < 60]
    if len(times) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded. Try again in a minute.")
    times.append(now)


def _error_message(exc: Exception) -> str:
    """Readable message from an exception, surfacing upstream LLM provider errors."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return f"LLM provider error: {err['message']}"
    msg = getattr(exc, "message", None)
    if isinstance(msg, str) and msg:
        return msg
    return f"{type(exc).__name__}: {exc}"


@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url.path}")
    return JSONResponse(status_code=500, content={"error": _error_message(exc)})


class ReviewRequest(BaseModel):
    question: str
    max_papers: int = 20
    language: str = "en"
    instructions: str = ""


class RefineRequest(BaseModel):
    instruction: str
    history: list[dict] = []


class ChatRequest(BaseModel):
    message: str


_state: PipelineState | None = None
_config: Config | None = None
_model = None
_chat_history: list[tuple[str, str]] = []


def _get_config():
    global _config, _model
    if _config is None:
        from dotenv import load_dotenv
        load_dotenv()
        _config = Config()
        _config.ensure_directories()
        _model = _build_model(_config)
    return _config, _model


_static_dir = Path(__file__).parent / "static"
_dist_dir = _static_dir / "dist"

if _dist_dir.exists():
    app.mount("/assets", StaticFiles(directory=_dist_dir / "assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
async def index():
    dist_index = _dist_dir / "index.html"
    if dist_index.exists():
        return dist_index.read_text()
    legacy = _static_dir / "index.html"
    if legacy.exists():
        return legacy.read_text()
    return "<h1>LitScribe API</h1><p>See /docs for API documentation.</p>"


@app.get("/favicon.svg", include_in_schema=False)
async def favicon():
    f = _dist_dir / "favicon.svg"
    if f.exists():
        return FileResponse(f, media_type="image/svg+xml")
    raise HTTPException(404, "Not found")


@app.get("/api/health")
async def health():
    import os
    has_key = bool(os.getenv("llm-key") or os.getenv("LLM_API_KEY"))
    has_base = bool(os.getenv("llm-location") or os.getenv("LLM_API_BASE"))
    has_model = bool(os.getenv("llm-model") or os.getenv("LLM_MODEL"))
    configured = has_key and has_base and has_model
    return {
        "configured": configured,
        "llm_key_set": has_key,
        "llm_base_set": has_base,
        "llm_model_set": has_model,
    }


class SetupRequest(BaseModel):
    api_key: str
    api_base: str
    model: str
    ncbi_email: str = ""
    ncbi_api_key: str = ""


@app.post("/api/setup")
async def setup(req: SetupRequest):
    import os
    env_path = Path(__file__).parents[2] / ".env"
    lines = [
        f"llm-key={req.api_key}",
        f"llm-location={req.api_base}",
        f"llm-model={req.model}",
    ]
    if req.ncbi_email:
        lines.append(f"NCBI_EMAIL={req.ncbi_email}")
    if req.ncbi_api_key:
        lines.append(f"NCBI_API_KEY={req.ncbi_api_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Reload env vars
    os.environ["llm-key"] = req.api_key
    os.environ["llm-location"] = req.api_base
    os.environ["llm-model"] = req.model
    global _config, _model
    _config = None
    _model = None
    return {"status": "ok"}


@app.post("/api/review")
async def run_review(req: ReviewRequest):
    await check_rate_limit("review")
    global _state
    config, model = _get_config()
    _state = PipelineState(
        research_question=req.question,
        language=req.language,
    )

    async def _run():
        from litscribe.tools.pipeline import (
            step_plan, step_search, step_read, step_contradictions,
            step_synthesize, step_ground, step_review,
        )
        t0 = time.time()

        yield _sse("status", {"step": "plan", "message": "Planning research..."})
        await step_plan(model, _state)
        yield _sse("plan", {
            "domain": _state.domain,
            "sub_topics": [st.name for st in _state.plan.sub_topics] if _state.plan else [],
        })

        yield _sse("status", {"step": "search", "message": "Searching papers..."})
        await step_search(model, _state, config, max_papers=req.max_papers)
        yield _sse("search", {
            "papers_found": len(_state.papers),
            "papers": [{
                "title": p.title,
                "authors": p.authors[:3],
                "year": p.year,
                "url": _paper_url(p),
            } for p in _state.papers[:20]],
        })

        yield _sse("status", {"step": "read", "message": f"Analyzing {len(_state.papers)} papers..."})
        await step_read(model, _state)
        yield _sse("read", {"analyzed": len(_state.analyses)})

        yield _sse("status", {"step": "contradictions", "message": "Detecting contradictions..."})
        await step_contradictions(model, _state)
        contra_count = _state.contradiction_report.count if _state.contradiction_report else 0
        if contra_count:
            yield _sse("contradictions", {"count": contra_count})

        # Build synthesis instructions with contradictions
        synth_instructions = req.instructions
        if contra_count > 0:
            from litscribe.tools.contradictions import format_contradictions_for_synthesis
            from litscribe.tools.cite_keys import assign_cite_keys
            key_map = assign_cite_keys(_state.papers)
            contra_text = format_contradictions_for_synthesis(_state.contradiction_report, key_map)
            synth_instructions += f"\n\nInclude a section discussing these contradictions:\n{contra_text}"

        yield _sse("status", {"step": "synthesize", "message": "Writing review..."})
        await step_synthesize(model, _state, synth_instructions)
        yield _sse("synthesis", {
            "word_count": _state.synthesis.word_count,
            "themes": [t.name for t in _state.synthesis.themes],
        })

        yield _sse("status", {"step": "ground", "message": "Verifying citations..."})
        await step_ground(model, _state)
        # Send grounding report to frontend
        if hasattr(_state, 'grounding_report') and _state.grounding_report:
            gr = _state.grounding_report
            yield _sse("grounding", {
                "total": gr.total_citations,
                "verified": gr.verified,
                "unsupported": gr.unsupported,
                "accuracy": round(gr.accuracy * 100),
            })

        yield _sse("status", {"step": "review", "message": "Evaluating quality..."})
        await step_review(model, _state)
        yield _sse("review", {
            "score": _state.assessment.score,
            "passed": _state.assessment.passed,
        })

        # Save session
        try:
            from litscribe.store.sessions import SessionStore
            store = SessionStore(config.db_path)
            sid = await store.save_session(_state)
            logger.info(f"Session saved: {sid}")
        except Exception as e:
            logger.warning(f"Session save failed: {e}")

        elapsed = time.time() - t0
        yield _sse("complete", {
            "text": _state.synthesis.text,
            "word_count": _state.synthesis.word_count,
            "score": _state.assessment.score,
            "papers": len(_state.papers),
            "time": round(elapsed, 1),
        })

    async def stream():
        try:
            async for chunk in _run():
                yield chunk
        except Exception as exc:
            logger.exception("Review pipeline failed")
            yield _sse("error", {"message": _error_message(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream")


class PlanRequest(BaseModel):
    question: str
    language: str = "en"


@app.post("/api/plan")
async def plan_review_skeleton(req: PlanRequest):
    """Phase 1 of review generation: research plan → proposed outline skeleton.

    The client shows the skeleton for confirmation, then generates the full
    review through /api/outline-review with the (possibly edited) outline.
    """
    await check_rate_limit("plan")
    global _state
    config, model = _get_config()
    from litscribe.tools.pipeline import step_plan

    _state = PipelineState(research_question=req.question, language=req.language)
    await step_plan(model, _state)

    topics = [st.name for st in _state.plan.sub_topics] if _state.plan else []
    zh = req.language == "zh"

    # The planner emits English sub-topic names even for Chinese questions
    # (search queries must be English). Localize the section titles so the
    # outline isn't a mix of 中文 + English headings.
    if zh and topics and any(re.search(r"[A-Za-z]", t) for t in topics):
        try:
            joined = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
            resp = await model.ainvoke(
                "把下面的文献综述章节标题翻译成简洁、专业的中文学术标题，"
                "保持顺序与编号，每行一个，只输出翻译结果：\n" + joined
            )
            lines = [l.strip() for l in resp.content.strip().split("\n") if l.strip()]
            zh_titles = [re.sub(r"^\d+[.、)]\s*", "", l) for l in lines]
            if len(zh_titles) == len(topics):
                topics = zh_titles
        except Exception as e:
            logger.warning(f"Title localization failed, keeping originals: {e}")

    titles = [
        "引言" if zh else "Introduction",
        *topics,
        "总结与展望" if zh else "Conclusion and Future Directions",
    ]
    return {
        "domain": _state.domain,
        "sections": [
            {"number": str(i + 1), "title": t, "level": 1, "enabled": True}
            for i, t in enumerate(titles)
        ],
        "outline_text": "\n".join(f"{i + 1} {t}" for i, t in enumerate(titles)),
    }


@app.post("/api/refine")
async def refine(req: RefineRequest):
    global _state
    await check_rate_limit("refine")
    logger.info(f"Refine request: '{req.instruction[:50]}', state={'has synthesis' if _state and _state.synthesis else 'NO synthesis'}")
    if _state is None or _state.synthesis is None:
        raise HTTPException(400, "No review to refine. Run a review first.")

    config, model = _get_config()
    from litscribe.tools.refinement import refine_review
    from litscribe.prompts.utils import format_summaries_for_prompt
    from litscribe.tools.synthesis import _enrich_analyses_with_papers

    papers_ctx = ""
    if _state.analyses:
        enriched = _enrich_analyses_with_papers(_state.analyses, _state.papers)
        papers_ctx = format_summaries_for_prompt(enriched, max_chars=5000)

    # Fold recent conversation into the instruction so follow-up edits
    # ("更正式一点", "改回上一版的语气") resolve against earlier turns.
    instruction = req.instruction
    if req.history:
        ctx = "\n".join(
            f"{h.get('role', '?')}: {h.get('content', '')}" for h in req.history[-6:]
        )
        instruction = (
            f"Earlier conversation (for context only):\n{ctx}\n\n"
            f"Current instruction: {req.instruction}"
        )

    old_text = _state.synthesis.text
    logger.info(f"Refine: current review {_state.synthesis.word_count} words, calling LLM...")
    new_review = await refine_review(
        model, _state.synthesis, instruction,
        _state.research_question, papers_ctx, _state.language,
        config=config,
    )
    _state.synthesis = new_review
    logger.info(f"Refine done: {new_review.word_count} words")

    from litscribe.tools.diff import diff_stats, unified_diff
    stats = diff_stats(old_text, new_review.text)
    diff = unified_diff(old_text, new_review.text, "before", "after")

    return {
        "text": new_review.text,
        "word_count": new_review.word_count,
        "diff": diff,
        "stats": stats,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    global _chat_history
    config, model = _get_config()
    from litscribe.agents import create_litscribe_agent

    agent, state, _ = create_litscribe_agent(config)

    messages = [(r, c) for r, c in _chat_history] + [("human", req.message)]
    result = await agent.ainvoke({"messages": messages})
    response = result["messages"][-1].content

    _chat_history.append(("human", req.message))
    _chat_history.append(("assistant", response))
    if len(_chat_history) > 20:
        _chat_history = _chat_history[-20:]

    return {"response": response}


@app.get("/api/state")
async def get_state():
    if _state is None:
        return {"status": "idle"}
    return {
        "question": _state.research_question,
        "papers": len(_state.papers),
        "analyses": len(_state.analyses),
        "has_synthesis": _state.synthesis is not None,
        "word_count": _state.synthesis.word_count if _state.synthesis else 0,
        "score": _state.assessment.score if _state.assessment else None,
    }


@app.get("/api/sessions")
async def list_sessions():
    config, _ = _get_config()
    from litscribe.store.sessions import SessionStore
    store = SessionStore(config.db_path)
    sessions = await store.list_sessions()
    logger.info(f"Sessions list: {len(sessions)} sessions, db={config.db_path}")
    return sessions


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    global _state
    config, _ = _get_config()
    from litscribe.store.sessions import SessionStore
    store = SessionStore(config.db_path)
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Load session into _state so refine works on it
    from litscribe.models.review import ReviewOutput
    if _state is None:
        _state = PipelineState()
    _state.research_question = session.get("research_question", "")
    _state.language = session.get("language", "en")
    _state.domain = session.get("domain", "")
    if session.get("review_text"):
        _state.synthesis = ReviewOutput(
            text=session["review_text"],
            citations=[], themes=[],
            word_count=session.get("word_count", 0),
            language=session.get("language", "en"),
        )
    logger.info(f"Session {session_id} loaded into state for refine")

    return session


class CommentRequest(BaseModel):
    session_id: str
    text: str
    section: str = ""
    author: str = "anonymous"


@app.post("/api/sessions/{session_id}/comments")
async def add_comment(session_id: str, req: CommentRequest):
    config, _ = _get_config()
    import aiosqlite
    db = await aiosqlite.connect(config.db_path)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            author TEXT DEFAULT 'anonymous',
            section TEXT DEFAULT '',
            text TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    await db.execute(
        "INSERT INTO comments (session_id, author, section, text) VALUES (?, ?, ?, ?)",
        (session_id, req.author, req.section, req.text),
    )
    await db.commit()
    await db.close()
    return {"status": "ok"}


@app.get("/api/sessions/{session_id}/comments")
async def get_comments(session_id: str):
    config, _ = _get_config()
    import aiosqlite
    db = await aiosqlite.connect(config.db_path)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            author TEXT DEFAULT 'anonymous',
            section TEXT DEFAULT '',
            text TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    rows = await db.execute_fetchall(
        "SELECT author, section, text, created_at FROM comments WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    )
    await db.close()
    return [{"author": r[0], "section": r[1], "text": r[2], "created_at": r[3]} for r in rows]


@app.get("/api/share/{session_id}")
async def share_session(session_id: str):
    config, _ = _get_config()
    from litscribe.store.sessions import SessionStore
    store = SessionStore(config.db_path)
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    from html import escape
    return HTMLResponse(f"""
    <html><head><title>LitScribe Review</title>
    <style>body{{font-family:sans-serif;max-width:800px;margin:40px auto;padding:0 20px;line-height:1.7}}
    pre{{white-space:pre-wrap}}</style></head>
    <body><h1>{escape(session['research_question'])}</h1>
    <p>Domain: {escape(session.get('domain',''))} | Papers: {session['papers_count']} | Score: {session['score']:.2f}</p>
    <hr><pre>{escape(session['review_text'])}</pre></body></html>
    """)


class DraftRequest(BaseModel):
    draft_text: str
    paper_texts: list[str] = []


class OutlineRequest(BaseModel):
    paper_texts: list[str]


@app.post("/api/draft-review")
async def api_draft_review(req: DraftRequest):
    await check_rate_limit("draft")
    config, model = _get_config()
    from litscribe.tools.local_review import review_draft, _extract_metadata
    from litscribe.models.paper import Paper
    from litscribe.models.analysis import PaperAnalysis
    from litscribe.prompts.reading import ABSTRACT_ONLY_ANALYSIS_PROMPT

    papers = []
    for i, text in enumerate(req.paper_texts):
        meta = await _extract_metadata(model, text, f"paper_{i}")
        papers.append(Paper(
            paper_id=f"local:{i}", title=meta.get("title", f"Paper {i}"),
            authors=meta.get("authors", []), abstract=meta.get("abstract", text[:300]),
            year=meta.get("year", 2024), sources={"local": str(i)},
        ))

    analyses = []
    for p in papers:
        prompt = ABSTRACT_ONLY_ANALYSIS_PROMPT.format(
            research_question="draft review", title=p.title,
            authors=", ".join(p.authors[:3]), year=p.year, venue="",
            abstract=p.abstract, metadata_section="",
        )
        try:
            r = await model.ainvoke(prompt)
            import json, re
            raw = r.content.strip()
            if raw.startswith("```"): raw = re.sub(r"^```\w*\n?", "", raw); raw = re.sub(r"\n?```$", "", raw)
            d = json.loads(raw)
            analyses.append(PaperAnalysis(
                paper_id=p.paper_id, key_findings=d.get("key_findings", []),
                methodology=d.get("methodology", ""), strengths=d.get("strengths", []),
                limitations=d.get("limitations", []), relevance_score=0.5, themes=[],
            ))
        except Exception as _e:
            logger.debug(f"Silent error: {_e}")

    return await review_draft(model, req.draft_text, papers, analyses)


@app.post("/api/outline")
async def api_outline(req: OutlineRequest):
    await check_rate_limit("outline")
    config, model = _get_config()
    from litscribe.tools.local_review import suggest_outline, _extract_metadata
    from litscribe.models.paper import Paper
    from litscribe.models.analysis import PaperAnalysis

    papers = []
    for i, text in enumerate(req.paper_texts):
        meta = await _extract_metadata(model, text, f"paper_{i}")
        papers.append(Paper(
            paper_id=f"local:{i}", title=meta.get("title", f"Paper {i}"),
            authors=meta.get("authors", []), abstract=meta.get("abstract", text[:300]),
            year=meta.get("year", 2024), sources={"local": str(i)},
        ))

    analyses = [PaperAnalysis(
        paper_id=p.paper_id, key_findings=[p.abstract[:200]],
        methodology="", strengths=[], limitations=[], relevance_score=0.5, themes=[],
    ) for p in papers]

    return await suggest_outline(model, papers, analyses)


@app.get("/api/citation-network")
async def get_citation_network():
    if not _state or not _state.papers:
        raise HTTPException(400, "No papers available")
    from litscribe.tools.analytics import build_citation_network, citation_network_to_mermaid
    from litscribe.tools.cite_keys import assign_cite_keys
    key_map = assign_cite_keys(_state.papers)
    network = build_citation_network(_state.papers, key_map)
    network["mermaid"] = citation_network_to_mermaid(network)
    return network


@app.get("/api/readability")
async def get_readability():
    if not _state or not _state.synthesis:
        raise HTTPException(400, "No review available")
    _, model = _get_config()
    from litscribe.tools.analytics import assess_readability
    return await assess_readability(model, _state.synthesis.text)


@app.get("/api/writing-analysis")
async def get_writing_analysis():
    if not _state or not _state.synthesis:
        raise HTTPException(400, "No review available")
    from litscribe.tools.analytics import analyze_writing
    return analyze_writing(_state.synthesis.text)


# ── Writing templates ──────────────────────────────────────

class TemplateApplyRequest(BaseModel):
    instructions: str = ""
    word_count: int = 800


class TemplateCreateRequest(BaseModel):
    id: str
    label: str
    prompt: str


@app.get("/api/templates")
async def list_templates_api():
    config, _ = _get_config()
    from litscribe.tools.templates_run import list_templates
    return {"templates": list_templates(config.data_dir)}


@app.post("/api/templates")
async def create_template_api(req: TemplateCreateRequest):
    config, _ = _get_config()
    from litscribe.tools.templates_run import load_custom, save_custom
    tid = re.sub(r"[^a-z0-9-]", "-", req.id.strip().lower()) or "custom"
    custom = load_custom(config.data_dir)
    custom[tid] = {"label": req.label.strip() or tid, "prompt": req.prompt}
    save_custom(config.data_dir, custom)
    return {"id": tid, "ok": True}


@app.delete("/api/templates/{template_id}")
async def delete_template_api(template_id: str):
    config, _ = _get_config()
    from litscribe.tools.templates_run import load_custom, save_custom
    custom = load_custom(config.data_dir)
    if template_id not in custom:
        raise HTTPException(404, "Custom template not found (built-ins can't be deleted)")
    del custom[template_id]
    save_custom(config.data_dir, custom)
    return {"ok": True}


@app.post("/api/templates/{template_id}/apply")
async def apply_template_api(template_id: str, req: TemplateApplyRequest):
    config, model = _get_config()
    from litscribe.tools.templates_run import apply_template, list_templates
    needs = next((t for t in list_templates(config.data_dir) if t["id"] == template_id), None)
    if needs is None:
        raise HTTPException(404, f"Unknown template: {template_id}")
    papers = _state.papers if (_state and needs["needs_papers"]) else []
    if needs["needs_papers"] and not papers:
        raise HTTPException(400, "This template needs a review's papers — run a review first.")
    text = await apply_template(
        model, template_id, papers, req.instructions, req.word_count, config.data_dir,
    )
    return {"text": text}


class MultiReviewRequest(BaseModel):
    session_ids: list[str]


@app.post("/api/compare-reviews")
async def api_compare_reviews(req: MultiReviewRequest):
    config, model = _get_config()
    from litscribe.store.sessions import SessionStore
    store = SessionStore(config.db_path)

    reviews = []
    for sid in req.session_ids:
        s = await store.get_session(sid)
        if s and s.get("review_text"):
            reviews.append(s["review_text"])

    if len(reviews) < 2:
        raise HTTPException(400, "Need at least 2 reviews to compare")

    from litscribe.tools.analytics import compare_reviews
    return await compare_reviews(model, reviews)


@app.get("/api/claims")
async def get_claims():
    if _state is None or _state.synthesis is None:
        raise HTTPException(400, "No review available")
    from litscribe.tools.claim_chain import build_claim_chain
    from litscribe.tools.cite_keys import assign_cite_keys
    key_map = assign_cite_keys(_state.papers) if _state.papers else {}
    report = getattr(_state, "grounding_report", None)
    return build_claim_chain(_state.synthesis.text, report, key_map)


@app.get("/api/export/{format}")
async def export(format: str, style: str = "apa"):
    if _state is None or _state.synthesis is None:
        raise HTTPException(400, "No review to export.")
    from litscribe.tools.export import export_review
    result = await export_review(_state.synthesis, _state.papers, format, style)
    return {"content": result.get("content", "")}


@app.post("/api/upload-outline")
async def upload_outline_file(file: UploadFile = File(...)):
    import tempfile, os
    suffix = Path(file.filename or "outline.txt").suffix.lower()
    if suffix not in (".docx", ".md", ".txt"):
        raise HTTPException(400, f"Unsupported format: {suffix}. Use .docx, .md, or .txt")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    try:
        from litscribe.tools.outline_parser import parse_outline, outline_to_sections

        roots = parse_outline(tmp.name)
        sections = outline_to_sections(roots)

        lines: list[str] = []
        def _walk(node, indent=0):
            prefix = f"{node.number} " if node.number else ""
            lines.append(f"{'  ' * indent}{prefix}{node.title}")
            for c in node.children:
                _walk(c, indent + 1)
        for r in roots:
            _walk(r)

        return {
            "filename": file.filename,
            "sections": [{"title": s["title"], "number": s["number"], "level": s["level"]} for s in sections],
            "text": "\n".join(lines),
            "total_sections": len(sections),
        }
    finally:
        os.unlink(tmp.name)


class OutlineReviewRequest(BaseModel):
    outline_text: str
    language: str = "en"
    max_papers_per_section: int = 10
    constraints: str = ""
    section_filter: str = ""


@app.post("/api/outline-review")
async def run_outline_review_api(req: OutlineReviewRequest):
    await check_rate_limit("outline-review")
    global _state
    config, model = _get_config()

    async def _run():
        from litscribe.tools.outline_review import run_outline_review
        import tempfile, os

        # Write outline to a temp .md file
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
        tmp.write(req.outline_text)
        tmp.close()

        # Bridge run_outline_review's sync on_progress callback into this
        # SSE generator so the client sees per-section search/read/write activity.
        queue: asyncio.Queue = asyncio.Queue()

        def on_progress(event: str, data: dict):
            queue.put_nowait((event, data))

        try:
            task = asyncio.create_task(run_outline_review(
                model, config, tmp.name,
                max_papers_per_section=req.max_papers_per_section,
                language=req.language,
                constraints=req.constraints,
                section_filter=req.section_filter or None,
                on_progress=on_progress,
            ))

            while not (task.done() and queue.empty()):
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=0.3)
                except asyncio.TimeoutError:
                    continue
                if event == "complete":
                    continue  # final complete event below carries the full text
                yield _sse(event, data)

            result = task.result()
            if result.get("error"):
                yield _sse("error", {"message": result["error"]})
                return

            # Populate global state so a follow-up /api/refine has a review to
            # work on — the skeleton-first flow generates via this endpoint, so
            # without this refine always 400'd ("No review to refine").
            global _state
            first_line = next((l.strip() for l in req.outline_text.splitlines() if l.strip()), "")
            topic = re.sub(r"^\d+[.、)\s]+", "", first_line) or "literature review"
            _state = PipelineState(research_question=topic, language=req.language)
            _state.synthesis = ReviewOutput(
                text=result.get("text", ""),
                word_count=result.get("total_words", 0),
                language=req.language,
            )

            yield _sse("complete", {
                "text": result.get("text", ""),
                "total_words": result.get("total_words", 0),
                "total_papers": result.get("total_papers", 0),
                "time": result.get("time", 0),
                "coverage": result.get("coverage"),
                "sections": [
                    {"title": s["title"], "words": s["word_count"], "papers": s["papers_count"]}
                    for s in result.get("sections", [])
                ],
            })
        finally:
            os.unlink(tmp.name)

    async def stream():
        try:
            async for chunk in _run():
                yield chunk
        except Exception as exc:
            logger.exception("Outline review failed")
            yield _sse("error", {"message": _error_message(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream")


def _paper_url(paper) -> str:
    if paper.doi:
        return f"https://doi.org/{paper.doi}"
    for source, sid in (paper.sources or {}).items():
        if source == "openalex":
            return f"https://openalex.org/{sid}"
        if source == "pubmed":
            return f"https://pubmed.ncbi.nlm.nih.gov/{sid}"
        if source == "europepmc":
            return f"https://europepmc.org/article/MED/{sid}"
        if source == "arxiv":
            return f"https://arxiv.org/abs/{sid}"
        if source == "s2":
            return f"https://www.semanticscholar.org/paper/{sid}"
        if source == "crossref" and sid.startswith("10."):
            return f"https://doi.org/{sid}"
    return ""


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

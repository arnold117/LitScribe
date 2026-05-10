from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from litscribe.config import Config
from litscribe.agents import _build_model
from litscribe.tools.status import PipelineState

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


class ReviewRequest(BaseModel):
    question: str
    max_papers: int = 20
    language: str = "en"
    instructions: str = ""


class RefineRequest(BaseModel):
    instruction: str


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


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>LitScribe API</h1><p>See /docs for API documentation.</p>"


@app.post("/api/review")
async def run_review(req: ReviewRequest):
    await check_rate_limit("review")
    global _state
    config, model = _get_config()
    _state = PipelineState(
        research_question=req.question,
        language=req.language,
    )

    async def stream():
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
            "titles": [p.title for p in _state.papers[:10]],
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

    return StreamingResponse(stream(), media_type="text/event-stream")


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

    old_text = _state.synthesis.text
    logger.info(f"Refine: current review {_state.synthesis.word_count} words, calling LLM...")
    new_review = await refine_review(
        model, _state.synthesis, req.instruction,
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
        except Exception:
            pass

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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

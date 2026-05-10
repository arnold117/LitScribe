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
    global _state
    config, model = _get_config()
    _state = PipelineState(
        research_question=req.question,
        language=req.language,
    )

    async def stream():
        from litscribe.tools.pipeline import (
            step_plan, step_search, step_read, step_synthesize,
            step_ground, step_review,
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

        yield _sse("status", {"step": "synthesize", "message": "Writing review..."})
        await step_synthesize(model, _state, req.instructions)
        yield _sse("synthesis", {
            "word_count": _state.synthesis.word_count,
            "themes": [t.name for t in _state.synthesis.themes],
        })

        yield _sse("status", {"step": "ground", "message": "Verifying citations..."})
        await step_ground(model, _state)

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

    logger.info(f"Refine: current review {_state.synthesis.word_count} words, calling LLM...")
    new_review = await refine_review(
        model, _state.synthesis, req.instruction,
        _state.research_question, papers_ctx, _state.language,
    )
    _state.synthesis = new_review
    logger.info(f"Refine done: {new_review.word_count} words")

    return {
        "text": new_review.text,
        "word_count": new_review.word_count,
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
    config, _ = _get_config()
    from litscribe.store.sessions import SessionStore
    store = SessionStore(config.db_path)
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@app.get("/api/export/{format}")
async def export(format: str, style: str = "apa"):
    if _state is None or _state.synthesis is None:
        raise HTTPException(400, "No review to export.")
    from litscribe.tools.export import export_review
    result = await export_review(_state.synthesis, _state.papers, format, style)
    return {"content": result.get("content", "")}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

"""LitScribe CLI — thin client that talks to the LitScribe server."""
from __future__ import annotations

import json
import sys
from typing import Optional

import typer
import httpx

app = typer.Typer(
    name="litscribe",
    help="LitScribe: self-evolving multi-agent literature review engine.",
    no_args_is_help=True,
)

_DEFAULT_BASE = "http://localhost:8000"


def _client(base_url: str = _DEFAULT_BASE) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=300.0)


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

@app.command()
def review(
    question: str = typer.Argument(..., help="The research question to review."),
    tier: str = typer.Option("standard", "--tier", "-t", help="Review tier: quick | standard | comprehensive"),
    max_papers: int = typer.Option(40, "--max-papers", "-n", help="Maximum number of papers to retrieve"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model override"),
    language: str = typer.Option("en", "--language", "-l", help="Output language code"),
    graphrag: bool = typer.Option(True, "--graphrag/--no-graphrag", help="Enable GraphRAG enrichment"),
    server: str = typer.Option(_DEFAULT_BASE, "--server", help="LitScribe server URL"),
) -> None:
    """Start a new literature review job on the server."""
    payload: dict = {
        "question": question,
        "tier": tier,
        "max_papers": max_papers,
        "language": language,
        "graphrag": graphrag,
    }
    if model:
        payload["model"] = model

    try:
        with _client(server) as client:
            resp = client.post("/api/reviews", json=payload)
            resp.raise_for_status()
            data = resp.json()
        typer.echo(f"Review started — ID: {data['review_id']}")
        typer.echo(f"Status: {data['status']}")
    except httpx.ConnectError:
        typer.echo(f"Cannot connect to server at {server}. Is it running?", err=True)
        raise typer.Exit(1)
    except httpx.HTTPStatusError as exc:
        typer.echo(f"Server error {exc.response.status_code}: {exc.response.text}", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

@app.command()
def sessions(
    server: str = typer.Option(_DEFAULT_BASE, "--server", help="LitScribe server URL"),
) -> None:
    """List all review sessions on the server."""
    try:
        with _client(server) as client:
            resp = client.get("/api/reviews")
            resp.raise_for_status()
            items = resp.json()
        if not items:
            typer.echo("No sessions found.")
            return
        for item in items:
            typer.echo(f"[{item['review_id']}] {item['status']:10s}  {item['question'][:60]}")
    except httpx.ConnectError:
        typer.echo(f"Cannot connect to server at {server}.", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------

@app.command()
def skills(
    action: str = typer.Argument("list", help="Action: list | update"),
    name: Optional[str] = typer.Argument(None, help="Skill name / ID (for update)"),
    server: str = typer.Option(_DEFAULT_BASE, "--server", help="LitScribe server URL"),
) -> None:
    """Manage procedural skills stored in memory."""
    try:
        with _client(server) as client:
            if action == "list":
                resp = client.get("/api/memory/skills")
                resp.raise_for_status()
                items = resp.json()
                if not items:
                    typer.echo("No skills found.")
                else:
                    for s in items:
                        typer.echo(json.dumps(s))
            elif action == "update" and name:
                resp = client.put(f"/api/memory/skills/{name}", json={})
                resp.raise_for_status()
                typer.echo(resp.json())
            else:
                typer.echo(f"Unknown action '{action}'. Use list or update.", err=True)
                raise typer.Exit(1)
    except httpx.ConnectError:
        typer.echo(f"Cannot connect to server at {server}.", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@app.command()
def config(
    key: Optional[str] = typer.Argument(None, help="Config key to read or set"),
    value: Optional[str] = typer.Argument(None, help="Value to set (omit to read)"),
) -> None:
    """Read or write LitScribe configuration values."""
    from litscribe.config import Config

    cfg = Config()
    if key is None:
        typer.echo(f"data_dir:        {cfg.data_dir}")
        typer.echo(f"skills_dir:      {cfg.skills_dir}")
        typer.echo(f"graphrag:        {cfg.graphrag_enabled}")
        typer.echo(f"default_model:   {cfg.llm.default_model}")
        typer.echo(f"api_base:        {cfg.llm.api_base}")
        return

    # Resolve key
    known = {
        "data_dir": lambda: str(cfg.data_dir),
        "skills_dir": lambda: str(cfg.skills_dir),
        "graphrag": lambda: str(cfg.graphrag_enabled),
        "default_model": lambda: cfg.llm.default_model,
        "api_base": lambda: cfg.llm.api_base,
    }
    if value is None:
        resolver = known.get(key)
        if resolver:
            typer.echo(resolver())
        else:
            typer.echo(f"Unknown config key: {key}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo(
            f"Config writes are not yet persisted. Would set {key}={value}",
            err=True,
        )
        raise typer.Exit(1)

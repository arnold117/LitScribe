from __future__ import annotations

import asyncio
import logging
import sys

import typer

app = typer.Typer(help="LitScribe — AI-powered literature review engine")


@app.command()
def chat(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Interactive chat with LitScribe."""
    asyncio.run(_chat_loop(verbose))


@app.command()
def review(
    question: str = typer.Argument(..., help="Research question"),
    max_papers: int = typer.Option(40, "--max-papers", "-n"),
    language: str = typer.Option("en", "--language", "-l"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run a literature review directly."""
    asyncio.run(_run_review(question, max_papers, language, verbose))


@app.command()
def sessions(session_id: str = typer.Argument(None, help="Session ID to view details, or omit to list all")):
    """List past reviews or view a specific session."""
    asyncio.run(_manage_sessions(session_id))


@app.command()
def skills(action: str = typer.Argument("list", help="list | show <slug> | delete <slug>")):
    """Manage learned research skills."""
    asyncio.run(_manage_skills(action))


@app.command()
def export(
    format: str = typer.Argument("markdown", help="markdown | bibtex | citations"),
    style: str = typer.Option("apa", "--style", "-s", help="Citation style: apa, mla, ieee, chicago"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Export the last review."""
    asyncio.run(_export_review(format, style, output))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Start the Web UI server."""
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    print(f"LitScribe Web UI: http://{host}:{port}")
    uvicorn.run("litscribe.api.app:app", host=host, port=port, reload=False)


async def _init_agent(verbose: bool = False):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    from dotenv import load_dotenv
    load_dotenv()

    from litscribe.config import Config
    from litscribe.agents import create_litscribe_agent

    config = Config()
    config.ensure_directories()

    memory = None
    try:
        from litscribe.evolution.memory_manager import MemoryManager
        memory = MemoryManager(config.db_path, config.chroma_path, config.skills_dir)
        await memory.initialize()
    except Exception as e:
        logging.warning(f"Memory init skipped: {e}")

    agent, state, token_mw = create_litscribe_agent(config, memory)
    return agent, state, token_mw, memory


async def _chat_loop(verbose: bool):
    agent, state, token_mw, memory = await _init_agent(verbose)

    print("LitScribe Chat (type 'exit' to quit)")
    print()

    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        try:
            messages = []
            for role, content in history:
                messages.append((role, content))
            messages.append(("human", user_input))

            result = await agent.ainvoke({"messages": messages})
            response = result["messages"][-1].content
            print(f"\nLitScribe> {response}\n")

            history.append(("human", user_input))
            history.append(("assistant", response))

            if len(history) > 20:
                history = history[-20:]
        except Exception as e:
            print(f"\nError: {e}\n")

    if token_mw.total_calls > 0:
        print(f"\nToken usage: {token_mw.summary()}")
    if memory:
        await memory.close()


async def _run_review(question: str, max_papers: int, language: str, verbose: bool):
    agent, state, token_mw, memory = await _init_agent(verbose)

    state.research_question = question
    state.language = language

    prompt = (
        f"Run a complete literature review on: {question}\n"
        f"Max papers: {max_papers}, Language: {language}\n"
        f"Start by calling check_pipeline_status, then follow the recommendation."
    )

    try:
        result = await agent.ainvoke({"messages": [("human", prompt)]})
        response = result["messages"][-1].content
        print(f"\n{response}\n")
    except Exception as e:
        print(f"\nReview failed: {e}")

    if token_mw.total_calls > 0:
        print(f"\nToken usage: {token_mw.summary()}")
    if memory:
        await memory.close()


async def _manage_sessions(session_id: str | None):
    from dotenv import load_dotenv
    load_dotenv()
    from litscribe.config import Config
    from litscribe.store.sessions import SessionStore

    config = Config()
    config.ensure_directories()
    store = SessionStore(config.db_path)

    if session_id is None:
        sessions = await store.list_sessions()
        if not sessions:
            print("No sessions yet. Run 'litscribe review <question>' to create one.")
            return
        print(f"{'ID':<10} {'Score':>5} {'Papers':>6} {'Words':>6} {'Question':<50} {'Date'}")
        print("-" * 95)
        for s in sessions:
            print(f"{s['session_id']:<10} {s['score']:>5.2f} {s['papers']:>6} {s['words']:>6} "
                  f"{s['question'][:50]:<50} {s['created_at'][:10]}")
    else:
        session = await store.get_session(session_id)
        if not session:
            print(f"Session '{session_id}' not found.")
            return
        print(f"Session: {session['session_id']}")
        print(f"Question: {session['research_question']}")
        print(f"Domain: {session['domain']}")
        print(f"Papers: {session['papers_count']}, Words: {session['word_count']}, Score: {session['score']:.2f}")
        print(f"Created: {session['created_at']}")
        print()

        versions = await store.get_versions(session_id)
        if versions:
            print(f"Versions ({len(versions)}):")
            for v in versions:
                instr = f" — {v['instruction']}" if v['instruction'] else ""
                print(f"  v{v['version']}: {v['words']} words{instr} ({v['created_at'][:16]})")

        print(f"\nReview (first 500 chars):")
        print(session['review_text'][:500])


async def _manage_skills(action: str):
    from dotenv import load_dotenv
    load_dotenv()

    from litscribe.config import Config
    config = Config()
    config.ensure_directories()

    try:
        from litscribe.evolution.memory_manager import MemoryManager
        memory = MemoryManager(config.db_path, config.chroma_path, config.skills_dir)
        await memory.initialize()

        if action == "list":
            skills = memory.procedural.list_skills()
            if not skills:
                print("No skills learned yet.")
            for s in skills:
                print(f"  [{s.get('slug', '?')}] {s.get('name', '?')} "
                      f"(v{s.get('version', 1)}, success={s.get('success_rate', 0):.2f})")
        else:
            print(f"Unknown action: {action}. Use: list")

        await memory.close()
    except Exception as e:
        print(f"Skills error: {e}")


async def _export_review(format: str, style: str, output: str | None):
    from dotenv import load_dotenv
    load_dotenv()

    from litscribe.config import Config
    from litscribe.tools.export import export_review
    from litscribe.models.review import ReviewOutput

    print(f"Export not yet connected to session storage.")
    print(f"Use 'litscribe chat' and ask the agent to export after a review.")


def main():
    app()


if __name__ == "__main__":
    main()

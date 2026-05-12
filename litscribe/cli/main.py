from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

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
def draft(
    draft_file: str = typer.Argument(..., help="Path to draft file (.md/.txt)"),
    papers: list[str] = typer.Argument(None, help="Paths to PDFs or BibTeX files"),
):
    """Analyze a draft review with reference papers. Suggests improvements."""
    asyncio.run(_review_draft(draft_file, papers or []))


@app.command()
def outline(
    papers: list[str] = typer.Argument(..., help="Paths to PDFs or BibTeX files"),
):
    """Given a collection of papers, suggest what review to write + what's missing."""
    asyncio.run(_suggest_outline(papers))


@app.command()
def augment(
    question: str = typer.Argument(..., help="Research question"),
    papers: list[str] = typer.Argument(None, help="Paths to local PDFs or BibTeX"),
    max_extra: int = typer.Option(10, "--max-extra", "-n"),
    language: str = typer.Option("en", "--language", "-l"),
):
    """Write a review using your papers + additional search results."""
    asyncio.run(_augmented_review(question, papers or [], max_extra, language))


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
    session_id: str = typer.Option(None, "--session", "-s", help="Session ID (default: latest)"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Export a review to file."""
    asyncio.run(_export_review(format, session_id, output))


@app.command()
def evaluate(
    max_papers: int = typer.Option(8, "--max-papers", "-n"),
    output: str = typer.Option(None, "--output", "-o", help="Save report to file"),
):
    """Run benchmark evaluation across multiple domains."""
    asyncio.run(_run_benchmark(max_papers, output))


@app.command()
def init():
    """Interactive setup wizard — configure LitScribe for first use."""
    _run_init()


@app.command(name="from-outline")
def from_outline(
    outline_file: str = typer.Argument(..., help="Path to outline file (.docx / .md / .txt)"),
    max_papers: int = typer.Option(10, "--max-papers", "-n", help="Max papers per section"),
    language: str = typer.Option("en", "--language", "-l"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    constraints: str = typer.Option("", "--constraints", "-c", help="Global constraints (e.g., species list)"),
    sections: str = typer.Option(None, "--sections", "-s", help="Run specific sections (e.g., '8.1' or '8.1,8.2')"),
    check_consistency: bool = typer.Option(False, "--check-consistency", help="Run cross-section consistency check"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate a literature review from a structured outline."""
    asyncio.run(_run_outline_review(
        outline_file, max_papers, language, output, verbose,
        constraints, sections, check_consistency,
    ))


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
    from litscribe.config import Config
    from litscribe.agents import _build_model
    from litscribe.tools.status import PipelineState
    from litscribe.tools.pipeline import (
        step_plan, step_search, step_read, step_contradictions,
        step_graphrag, step_synthesize, step_ground, step_review,
    )
    from dotenv import load_dotenv
    load_dotenv()

    if verbose:
        logging.basicConfig(level=logging.INFO)

    config = Config()
    config.ensure_directories()
    model = _build_model(config)
    state = PipelineState(research_question=question, language=language)

    memory = None
    try:
        from litscribe.evolution.memory_manager import MemoryManager
        memory = MemoryManager(config.db_path, config.chroma_path, config.skills_dir)
        await memory.initialize()
    except Exception as _e:
        logger.debug(f"Silent error: {_e}")

    import time
    t = time.time()

    print(f"Planning...")
    await step_plan(model, state)
    print(f"  Domain: {state.domain}, {len(state.plan.sub_topics)} sub-topics")

    print(f"Searching papers...")
    await step_search(model, state, config, max_papers)
    print(f"  Found {len(state.papers)} papers")

    print(f"Analyzing papers...")
    await step_read(model, state)
    print(f"  Analyzed {len(state.analyses)} papers")

    print(f"Detecting contradictions...")
    await step_contradictions(model, state)
    if state.contradiction_report and state.contradiction_report.count:
        print(f"  Found {state.contradiction_report.count} contradictions")

    print(f"Building knowledge graph...")
    await step_graphrag(model, state)

    print(f"Writing review...")
    await step_synthesize(model, state)
    print(f"  {state.synthesis.word_count} words, {len(state.synthesis.themes)} themes")

    print(f"Debating (reviewer ↔ synthesizer)...")
    from litscribe.tools.pipeline import step_debate
    await step_debate(model, state)

    print(f"Verifying citations...")
    await step_ground(model, state)
    if hasattr(state, "grounding_report") and state.grounding_report:
        gr = state.grounding_report
        print(f"  {gr.verified}/{gr.total_citations} verified ({gr.accuracy:.0%} accuracy), {gr.unsupported} auto-fixed")

    print(f"Evaluating quality...")
    await step_review(model, state)
    print(f"  Score: {state.assessment.score:.2f}, Coverage: {state.assessment.coverage_score:.2f}")

    # Save session
    session_id = ""
    try:
        from litscribe.store.sessions import SessionStore
        store = SessionStore(config.db_path)
        session_id = await store.save_session(state)
    except Exception as _e:
        logger.debug(f"Silent error: {_e}")

    elapsed = time.time() - t
    print(f"\n{'='*60}")
    print(f"Review complete in {elapsed:.0f}s")
    print(f"  Papers: {len(state.papers)} | Analyzed: {len(state.analyses)}")
    print(f"  Words: {state.synthesis.word_count} | Themes: {len(state.synthesis.themes)}")
    print(f"  Score: {state.assessment.score:.2f} | Coverage: {state.assessment.coverage_score:.2f}")
    if state.contradiction_report and state.contradiction_report.count:
        print(f"  Contradictions: {state.contradiction_report.count}")
    if session_id:
        print(f"  Session: {session_id}")
    from litscribe.tools.output import save_review
    filepath = save_review(state.synthesis.text, question)
    print(f"  📄 File: {filepath}")
    print(f"{'='*60}\n")
    print(state.synthesis.text[:3000])
    if len(state.synthesis.text) > 3000:
        print(f"\n... ({state.synthesis.word_count} words total, see {filename} for full text)")

    if memory:
        await memory.close()


async def _review_draft(draft_file: str, paper_paths: list[str]):
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    from litscribe.config import Config
    from litscribe.agents import _build_model
    from litscribe.tools.local_review import parse_local_papers, review_draft
    from litscribe.tools.pipeline import step_read

    config = Config(); config.ensure_directories()
    model = _build_model(config)

    draft_text = Path(draft_file).read_text(encoding="utf-8")
    print(f"Draft: {len(draft_text)} chars")

    if paper_paths:
        print(f"Parsing {len(paper_paths)} reference files...")
        papers = await parse_local_papers(model, paper_paths)
    else:
        papers = []

    print(f"Analyzing {len(papers)} papers...")
    from litscribe.tools.status import PipelineState
    state = PipelineState(research_question="draft review")
    state.papers = papers
    await step_read(model, state)

    print("Reviewing draft...")
    result = await review_draft(model, draft_text, papers, state.analyses)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n{'='*60}")
    print("STRENGTHS:")
    for s in result.get("strengths", []): print(f"  + {s}")
    print("\nWEAKNESSES:")
    for w in result.get("weaknesses", []): print(f"  - {w.get('issue','')}: {w.get('suggestion','')}")
    print("\nMISSING TOPICS:")
    for m in result.get("missing_topics", []): print(f"  ? {m}")
    print("\nSUGGESTED ADDITIONS:")
    for a in result.get("suggested_additions", []): print(f"  + [{a.get('paper','')}] → {a.get('where','')}: {a.get('why','')}")
    print("\nREVISED OUTLINE:")
    for o in result.get("revised_outline", []): print(f"  {o}")


async def _suggest_outline(paper_paths: list[str]):
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    from litscribe.config import Config
    from litscribe.agents import _build_model
    from litscribe.tools.local_review import parse_local_papers, suggest_outline
    from litscribe.tools.pipeline import step_read

    config = Config(); config.ensure_directories()
    model = _build_model(config)

    print(f"Parsing {len(paper_paths)} files...")
    papers = await parse_local_papers(model, paper_paths)

    from litscribe.tools.status import PipelineState
    state = PipelineState(research_question="outline suggestion")
    state.papers = papers
    await step_read(model, state)

    print("Analyzing themes and gaps...")
    result = await suggest_outline(model, papers, state.analyses)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f"SUGGESTED QUESTION: {result.get('suggested_question', '?')}")
    print("\nTHEMES:")
    for t in result.get("themes", []): print(f"  [{', '.join(t.get('papers',[])[:3])}] {t.get('name','')}: {t.get('description','')[:60]}")
    print("\nPROPOSED OUTLINE:")
    for o in result.get("proposed_outline", []): print(f"  {o}")
    print("\nGAPS (need more papers on):")
    for g in result.get("gaps", []): print(f"  ? {g}")
    print("\nSUGGESTED SEARCHES:")
    for q in result.get("search_queries", []): print(f"  → {q}")


async def _augmented_review(question: str, paper_paths: list[str], max_extra: int, language: str):
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    from litscribe.config import Config
    from litscribe.agents import _build_model
    from litscribe.tools.local_review import parse_local_papers, augmented_review

    config = Config(); config.ensure_directories()
    model = _build_model(config)

    local_papers = []
    if paper_paths:
        print(f"Parsing {len(paper_paths)} local files...")
        local_papers = await parse_local_papers(model, paper_paths)

    print(f"Running augmented review: {len(local_papers)} local + up to {max_extra} searched...")
    result = await augmented_review(model, config, question, local_papers, max_extra, language)
    print(result)


async def _run_benchmark(max_papers: int, output: str | None):
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    from litscribe.config import Config
    from litscribe.agents import _build_model
    from litscribe.tools.benchmark import run_benchmark, format_benchmark_report

    config = Config()
    config.ensure_directories()
    model = _build_model(config)

    print("Running benchmark (5 domains)...\n")
    results = await run_benchmark(config, model, max_papers=max_papers)

    report = format_benchmark_report(results)
    print(report)

    if output:
        from pathlib import Path
        Path(output).write_text(report, encoding="utf-8")
        print(f"\nReport saved to {output}")


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


def _run_init():
    env_path = Path(".env")
    if env_path.exists():
        overwrite = input(".env already exists. Overwrite? [y/N] ").strip().lower()
        if overwrite != "y":
            print("Aborted.")
            return

    print("=" * 50)
    print("LitScribe Setup")
    print("=" * 50)
    print()

    print("LitScribe works with any OpenAI-compatible API.")
    print("Examples: DeepSeek, Alibaba DashScope, OpenAI, Ollama, etc.")
    print()

    presets = {
        "1": ("DeepSeek", "https://api.deepseek.com/", "deepseek-chat"),
        "2": ("Alibaba DashScope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
        "3": ("OpenAI", "https://api.openai.com/v1", "gpt-4o"),
        "4": ("Ollama (local)", "http://localhost:11434/v1", "llama3"),
    }

    print("Choose a provider:")
    for k, (name, url, model) in presets.items():
        print(f"  [{k}] {name} ({url})")
    print("  [5] Custom")
    print()

    choice = input("Provider [1]: ").strip() or "1"

    if choice in presets:
        _, api_base, default_model = presets[choice]
    else:
        api_base = input("API base URL: ").strip()
        default_model = ""

    api_key = input("API key (paste, won't echo): ").strip()
    if not api_key:
        print("Warning: no API key set. You can add it to .env later.")
        api_key = "sk-your-key"

    model = input(f"Model name [{default_model}]: ").strip() or default_model

    print()
    print("Optional: academic search APIs (press Enter to skip)")
    ncbi_email = input("  NCBI email (for PubMed): ").strip()
    ncbi_key = input("  NCBI API key: ").strip()

    lines = [
        "# LitScribe config (generated by litscribe init)",
        f"llm-key={api_key}",
        f"llm-location={api_base}",
        f"llm-model={model}",
    ]
    if ncbi_email:
        lines.append(f"NCBI_EMAIL={ncbi_email}")
    if ncbi_key:
        lines.append(f"NCBI_API_KEY={ncbi_key}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"Config saved to {env_path.resolve()}")
    print()
    print("Next steps:")
    print("  litscribe chat              # interactive mode")
    print("  litscribe review 'topic'    # direct review")
    print("  litscribe serve             # web UI at localhost:8000")
    print("  litscribe from-outline f.docx  # review from outline")


async def _run_outline_review(
    outline_file: str, max_papers: int, language: str, output: str | None,
    verbose: bool, constraints: str = "", section_filter: str | None = None,
    check_consistency: bool = False,
):
    from dotenv import load_dotenv
    load_dotenv()

    if verbose:
        logging.basicConfig(level=logging.INFO)

    from litscribe.config import Config
    from litscribe.agents import _build_model
    from litscribe.tools.outline_review import run_outline_review, check_cross_section_consistency
    from litscribe.tools.outline_parser import parse_outline, outline_to_sections

    config = Config()
    config.ensure_directories()
    model = _build_model(config)

    roots = parse_outline(outline_file)
    sections = outline_to_sections(roots)
    print(f"Outline: {len(sections)} sections from {outline_file}")
    for i, s in enumerate(sections):
        indent = "  " * (s["level"])
        print(f"  {indent}{s.get('number', '')} {s['title']}")

    if constraints:
        print(f"\nConstraints: {constraints}")
    if section_filter:
        print(f"Section filter: {section_filter}")
    print()

    def on_progress(event: str, data: dict):
        if event == "section_start":
            print(f"[{data['index']+1}/{data['total']}] {data['title']}...")
        elif event == "section_done":
            print(f"  -> {data['papers']} papers, {data['words']} words")
        elif event == "assembling":
            print(f"\nAssembling {data['sections']} sections...")
        elif event == "coverage":
            cov = data
            print(f"\nCoverage: {cov['covered']}/{cov['total']} entities ({cov['coverage_pct']}%)")
            if cov.get("missing_entities"):
                print(f"  Missing: {', '.join(cov['missing_entities'][:10])}")
        elif event == "complete":
            print(f"\nDone in {data['time']}s: {data['total_words']} words, {data['total_papers']} papers")

    result = await run_outline_review(
        model, config, outline_file,
        max_papers_per_section=max_papers,
        language=language,
        constraints=constraints,
        section_filter=section_filter,
        on_progress=on_progress,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    if check_consistency and constraints:
        from litscribe.tools.outline_review import _parse_constraint_entities
        entities = _parse_constraint_entities(constraints)
        if entities:
            print("\nChecking cross-section consistency...")
            issues = await check_cross_section_consistency(
                model, result["sections"], entities,
            )
            if issues:
                print(f"  Found {len(issues)} inconsistencies:")
                for iss in issues:
                    print(f"    [{', '.join(iss['sections'])}] {iss['entity']}: {iss['issue']}")
                    if iss.get("suggestion"):
                        print(f"      Suggestion: {iss['suggestion']}")
            else:
                print("  No inconsistencies found.")

    out_path = output or f"outline_review_{Path(outline_file).stem}.md"
    Path(out_path).write_text(result["text"], encoding="utf-8")
    print(f"\nSaved to {out_path}")


async def _export_review(format: str, session_id: str | None, output: str | None):
    from dotenv import load_dotenv
    load_dotenv()
    from litscribe.config import Config
    from litscribe.store.sessions import SessionStore

    config = Config()
    config.ensure_directories()
    store = SessionStore(config.db_path)

    if session_id:
        session = await store.get_session(session_id)
        if not session:
            print(f"Session '{session_id}' not found.")
            return
    else:
        sessions = await store.list_sessions()
        if not sessions:
            print("No sessions. Run 'litscribe review <question>' first.")
            return
        session = await store.get_session(sessions[0]["session_id"])

    content = session["review_text"]
    filename = f"review_{session['session_id']}"

    if format == "bibtex":
        print("BibTeX export requires paper data. Use 'litscribe chat' or Web UI.")
        return

    if output:
        Path(output).write_text(content, encoding="utf-8")
        print(f"Exported to {output}")
    else:
        default_path = f"{filename}.md"
        Path(default_path).write_text(content, encoding="utf-8")
        print(f"Exported to {default_path}")
        print(f"  Session: {session['session_id']}")
        print(f"  Question: {session['research_question'][:60]}")
        print(f"  Words: {session['word_count']}, Score: {session['score']:.2f}")


def main():
    app()


if __name__ == "__main__":
    main()

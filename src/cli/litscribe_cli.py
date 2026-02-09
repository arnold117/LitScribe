#!/usr/bin/env python3
"""LitScribe CLI - Command line interface for academic literature search and analysis."""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cli.output import OutputManager, get_output


def detect_query_language(text: str) -> str:
    """Detect whether the query is primarily Chinese or English.

    Args:
        text: The research question text

    Returns:
        "zh" if Chinese characters dominate, "en" otherwise
    """
    chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    alpha = sum(1 for c in text if c.isalpha() or '\u4e00' <= c <= '\u9fff')
    if alpha > 0 and chinese / alpha > 0.3:
        return "zh"
    return "en"


def setup_logging(
    log_dir: Optional[Path] = None,
    verbose: bool = False,
    log_file_name: Optional[str] = None,
) -> Path:
    """Setup logging to both console and file.

    Args:
        log_dir: Directory for log files (default: logs/)
        verbose: If True, set DEBUG level; otherwise INFO
        log_file_name: Custom log file name (default: auto-generated with timestamp)

    Returns:
        Path to the log file
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log file name
    if log_file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"litscribe_{timestamp}.log"
    log_file = log_dir / log_file_name

    # Set log level
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # File handler (always DEBUG to capture everything)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logging.info(f"Logging initialized. Log file: {log_file}")
    return log_file


def print_header(title: str, out: Optional[OutputManager] = None) -> None:
    """Print a formatted header.

    Args:
        title: Header title
        out: Optional OutputManager (uses print only if None)
    """
    if out:
        out.header(title)
    else:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")


def print_paper(
    paper: dict,
    index: int,
    verbose: bool = False,
    out: Optional[OutputManager] = None,
) -> None:
    """Print a paper summary.

    Args:
        paper: Paper dictionary
        index: Paper index
        verbose: Show detailed info
        out: Optional OutputManager (uses print only if None)
    """
    if out:
        out.paper(paper, index, verbose)
    else:
        title = paper.get("title", "Unknown")
        if len(title) > 70:
            title = title[:67] + "..."

        print(f"{index}. {title}")

        authors = paper.get("authors", [])
        if authors:
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " ..."
            print(f"   Authors: {author_str}")

        year = paper.get("year", "N/A")
        citations = paper.get("citations", 0)
        sources = list(paper.get("sources", {}).keys())
        print(f"   Year: {year} | Citations: {citations} | Sources: {sources}")

        if verbose:
            if paper.get("doi"):
                print(f"   DOI: {paper['doi']}")
            if paper.get("arxiv_id"):
                print(f"   arXiv: {paper['arxiv_id']}")
            if paper.get("pmid"):
                print(f"   PMID: {paper['pmid']}")
            if paper.get("abstract"):
                abstract = paper["abstract"][:200] + "..." if len(paper.get("abstract", "")) > 200 else paper.get("abstract", "")
                print(f"   Abstract: {abstract}")

        print()


async def cmd_search(args) -> int:
    """Execute search command."""
    from aggregators.unified_search import search_all_sources

    out = get_output("litscribe.search")
    sources = args.sources.split(",") if args.sources else ["arxiv", "semantic_scholar", "pubmed"]

    out.header(f"Searching: '{args.query}'")
    out.stat("Sources", sources)
    out.stat("Max results per source", args.limit)
    out.stat("Sort by", args.sort)
    out.blank()

    try:
        result = await search_all_sources(
            query=args.query,
            sources=sources,
            max_per_source=args.limit,
            deduplicate=not args.no_dedup,
            sort_by=args.sort,
        )

        out.stat("Results per source", result["source_counts"])
        out.stat("Total before dedup", result["total_before_dedup"])
        out.stat("Total after dedup", result["total_after_dedup"])
        out.blank()

        papers = result["papers"]
        if not papers:
            out.info("No papers found.")
            return 0

        out.header("Results")
        for i, paper in enumerate(papers[: args.limit], 1):
            out.paper(paper, i, verbose=args.verbose)

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            out.success(f"Results saved to: {output_path}")

        return 0

    except Exception as e:
        out.error(f"Error: {e}")
        return 1


async def cmd_parse(args) -> int:
    """Execute parse command."""
    from mcp_servers.pdf_parser_server import parse_pdf

    out = get_output("litscribe.parse")
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        out.error(f"File not found: {pdf_path}")
        return 1

    out.header(f"Parsing: {pdf_path.name}")

    try:
        result = await parse_pdf(str(pdf_path))

        if "error" in result:
            out.error(result["error"])
            return 1

        out.stat("Title", result.get("title", "Unknown"))
        out.stat("Pages", result.get("pages", "N/A"))
        out.stat("Sections", len(result.get("sections", [])))
        out.stat("Tables", len(result.get("tables", [])))
        out.stat("References", len(result.get("references", [])))

        # Show sections
        if args.verbose and result.get("sections"):
            out.subheader("Sections")
            for sec in result["sections"][:10]:
                out.bullet(f"{sec.get('title', 'Untitled')} (p.{sec.get('start_page', '?')})")

        # Save markdown output
        if args.output:
            output_path = Path(args.output)
            content = result.get("markdown", "")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            out.success(f"Markdown saved to: {output_path}")
        else:
            # Print first 500 chars of markdown
            markdown = result.get("markdown", "")
            if markdown:
                out.preview("Preview", markdown, max_chars=500)

        return 0

    except Exception as e:
        out.error(f"Error: {e}")
        return 1


async def cmd_paper(args) -> int:
    """Get detailed information about a specific paper."""
    from mcp_servers.semantic_scholar_server import get_paper

    out = get_output("litscribe.paper")
    out.header(f"Fetching: {args.paper_id}")

    try:
        result = await get_paper(args.paper_id)

        if "error" in result:
            out.error(result["error"])
            return 1

        out.stat("Title", result.get("title", "Unknown"))
        out.stat("Authors", ", ".join(result.get("authors", [])[:5]))
        out.stat("Year", result.get("year", "N/A"))
        out.stat("Venue", result.get("venue", "N/A"))
        out.stat("Citations", result.get("citation_count", 0))
        out.stat("References", result.get("reference_count", 0))

        if result.get("doi"):
            out.stat("DOI", result["doi"])
        if result.get("arxiv_id"):
            out.stat("arXiv", result["arxiv_id"])
        if result.get("pdf_url"):
            out.stat("PDF", result["pdf_url"])

        if result.get("tldr"):
            out.subheader("TL;DR")
            out.info(f"   {result['tldr']}")

        if args.verbose:
            if result.get("abstract"):
                out.subheader("Abstract")
                out.info(result["abstract"])

            if result.get("top_citations"):
                out.subheader(f"Top Citations ({len(result['top_citations'])})")
                for c in result["top_citations"][:5]:
                    out.bullet(f"{c.get('title', 'Untitled')[:60]}...")

            if result.get("top_references"):
                out.subheader(f"Top References ({len(result['top_references'])})")
                for r in result["top_references"][:5]:
                    out.bullet(f"{r.get('title', 'Untitled')[:60]}...")

        # Save to file
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            out.success(f"Saved to: {output_path}")

        return 0

    except Exception as e:
        out.error(f"Error: {e}")
        return 1


async def cmd_citations(args) -> int:
    """Get citations for a paper."""
    from mcp_servers.semantic_scholar_server import get_paper_citations

    out = get_output("litscribe.citations")
    out.header(f"Citations for: {args.paper_id}")

    try:
        result = await get_paper_citations(args.paper_id, limit=args.limit)

        if "error" in result:
            out.error(result["error"])
            return 1

        out.stat("Total citations", result.get("total", 0))
        out.stat("Showing", result.get("count", 0))
        out.blank()

        for i, paper in enumerate(result.get("citations", []), 1):
            title = paper.get("title", "Unknown")[:60]
            year = paper.get("year", "N/A")
            cites = paper.get("citation_count", 0)
            out.info(f"{i}. [{year}] {title}... ({cites} cites)")

        return 0

    except Exception as e:
        out.error(f"Error: {e}")
        return 1


async def cmd_review(args) -> int:
    """Run complete literature review using multi-agent system."""
    import warnings
    from agents.graph import run_literature_review, run_refinement

    # Setup logging (to console and file)
    log_file = setup_logging(verbose=args.verbose)

    # Create OutputManager for unified output
    out = get_output("litscribe.review")

    # Suppress Pydantic serialization warnings from LiteLLM
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    research_question = args.question
    max_papers = min(args.papers, 500)  # Cap at 500
    sources = args.sources.split(",") if args.sources else ["arxiv", "semantic_scholar", "pubmed"]
    review_type = args.type

    # GraphRAG settings (Phase 7.5)
    graphrag_enabled = not getattr(args, "disable_graphrag", False)
    batch_size = getattr(args, "batch_size", 20)

    # Ablation flags (Phase 9.5)
    disable_self_review = getattr(args, "disable_self_review", False)
    disable_domain_filter = getattr(args, "disable_domain_filter", False)
    disable_snowball = getattr(args, "disable_snowball", False)

    # Default output: always save to output/ directory
    if args.output:
        output_path = Path(args.output)
    else:
        # Auto-generate output path based on timestamp
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize question for filename
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in research_question[:30])
        safe_name = safe_name.strip().replace(" ", "_")
        output_path = output_dir / f"review_{safe_name}_{timestamp}"

    out.header("LitScribe Literature Review")
    out.stat("Research Question", research_question)
    out.stat("Max Papers", max_papers)
    out.stat("Sources", sources)
    out.stat("Review Type", review_type)
    language = getattr(args, "lang", "en") or "en"

    # Language mismatch detection (Emergency Fix Step 5)
    query_lang = detect_query_language(research_question)
    if query_lang == "en" and language == "zh":
        out.warning("Query appears to be English but output language is set to Chinese (--lang zh).")
        confirm = input("  Continue with Chinese output? (Y/n): ").strip()
        if confirm.lower() == "n":
            language = "en"
            out.info("Switched output language to English.")
    elif query_lang == "zh" and language == "en":
        out.warning("Query appears to be Chinese but output language is English.")
        confirm = input("  Switch to Chinese output? (Y/n): ").strip()
        if confirm.lower() != "n":
            language = "zh"
            out.info("Switched output language to Chinese.")

    # Pre-generate thread_id so user can note it for resume
    import uuid as _uuid
    thread_id = str(_uuid.uuid4())

    out.stat("GraphRAG", "enabled" if graphrag_enabled else "disabled")
    out.stat("Language", language)
    out.stat("Output", output_path)
    out.stat("Thread ID", f"{thread_id}  (for resume if interrupted)")
    out.stat("Log File", log_file)
    out.blank()
    out.info("Starting multi-agent workflow...")
    out.bullet("Planning Agent: Complexity assessment & sub-topic decomposition")
    out.bullet("Discovery Agent: Searching and selecting papers")
    out.bullet("Critical Reading Agent: Analyzing papers")
    if graphrag_enabled:
        out.bullet("GraphRAG Agent: Building knowledge graph")
        out.bullet("Synthesis Agent: Generating review")
    else:
        out.bullet("Synthesis Agent: Generating review")
    out.bullet("Self-Review Agent: Quality assessment")
    out.blank()

    # Handle --plan-only: run planning agent only (Phase 9.2)
    plan_only = getattr(args, "plan_only", False)
    if plan_only:
        from agents.planning_agent import assess_and_decompose, format_plan_for_user
        try:
            out.info("Running planning agent only...")
            plan = await assess_and_decompose(research_question, max_papers=max_papers)
            out.subheader("Research Plan", "📋")
            out.info(format_plan_for_user(plan))
            out.blank()
            out.info("Use 'litscribe review' without --plan-only to execute the full workflow.")
            return 0
        except Exception as e:
            out.error(f"Planning failed: {e}")
            return 1

    try:
        # Local files workflow (Emergency Fix Step 6)
        local_files = getattr(args, "local_files", []) or []
        if local_files:
            out.info(f"Found {len(local_files)} local PDF file(s) to analyze.")
            choice = input("  Also search online for additional papers? (y/N): ").strip()
            if choice.lower() != "y":
                sources = []  # Empty sources = skip online discovery
                out.info("Will analyze local files only (no online search).")
            else:
                out.info("Will analyze local files AND search online.")

        # Planning confirmation (Emergency Fix Step 7a)
        # Run planning agent first; if complex, show plan and ask for confirmation
        injected_plan = None
        from agents.planning_agent import assess_and_decompose, format_plan_for_user
        try:
            out.info("Assessing research complexity...")
            plan = await assess_and_decompose(research_question, max_papers=max_papers)
            complexity = plan.get("complexity_score", 1)
            if complexity >= 3:
                out.subheader("Research Plan", "📋")
                out.info(format_plan_for_user(plan))
                out.blank()
                confirm = input("Proceed with this plan? (Y/n): ").strip()
                if confirm.lower() == "n":
                    out.info("Review cancelled.")
                    return 0
            injected_plan = plan
        except Exception as e:
            out.warning(f"Planning pre-check failed ({e}), will plan during workflow.")

        # Run the multi-agent workflow
        final_state = await run_literature_review(
            research_question=research_question,
            max_papers=max_papers,
            sources=sources,
            review_type=review_type,
            verbose=args.verbose,
            graphrag_enabled=graphrag_enabled,
            batch_size=batch_size,
            local_files=local_files,
            language=language,
            thread_id=thread_id,
            research_plan=injected_plan,
            disable_self_review=disable_self_review,
            disable_domain_filter=disable_domain_filter,
            disable_snowball=disable_snowball,
        )

        # Check for errors
        errors = final_state.get("errors", [])
        if errors:
            out.subheader("Warnings/Errors during processing", "⚠")
            for err in errors[-5:]:
                out.bullet(str(err))

        # Display research plan (Phase 9.2)
        research_plan = final_state.get("research_plan")
        if research_plan:
            from agents.planning_agent import format_plan_for_user
            out.subheader("Research Plan", "📋")
            out.stat("Complexity", f"{research_plan.get('complexity_score', '?')}/5")
            out.stat("Sub-topics", len(research_plan.get("sub_topics", [])))
            out.stat("Scope", research_plan.get("scope_estimate", "N/A"))
            if args.verbose:
                out.info(format_plan_for_user(research_plan))

        # Display results
        search_results = final_state.get("search_results")
        if search_results:
            out.subheader("Papers Found", "📚")
            out.stat("Total Found", search_results.get("total_found", 0))
            out.stat("Expanded Queries", len(search_results.get("expanded_queries", [])))
            out.stat("Source Distribution", search_results.get("source_counts", {}))

        analyzed = final_state.get("analyzed_papers", [])
        out.subheader("Papers Analyzed", "📖")
        out.stat("Count", len(analyzed))

        # Display GraphRAG results (Phase 7.5)
        knowledge_graph = final_state.get("knowledge_graph")
        if knowledge_graph:
            stats = knowledge_graph.get("stats", {})
            out.subheader("Knowledge Graph", "🕸")
            out.stat("Entities", stats.get("entity_count", 0))
            out.stat("Communities", stats.get("total_communities", 0))
            if stats.get("entity_types"):
                out.stat("Entity Types", stats["entity_types"])

        synthesis = final_state.get("synthesis")
        if synthesis:
            out.subheader("Review Generated", "✨")
            out.stat("Themes", len(synthesis.get("themes", [])))
            out.stat("Research Gaps", len(synthesis.get("gaps", [])))
            out.stat("Word Count", synthesis.get("word_count", 0))
            out.stat("Papers Cited", synthesis.get("papers_cited", 0))

            # Show themes
            if synthesis.get("themes"):
                out.subheader("Identified Themes", "📊")
                for i, theme in enumerate(synthesis["themes"], 1):
                    theme_name = theme.get("theme", "Unknown")
                    out.stat(f"{i}", theme_name)
                    if args.verbose and theme.get("description"):
                        out.bullet(theme["description"][:100] + "...")

            # Show gaps
            if synthesis.get("gaps"):
                out.subheader("Research Gaps", "🔍")
                for gap in synthesis["gaps"][:3]:
                    out.bullet(gap[:80] + "..." if len(gap) > 80 else gap)

        # Display self-review results (Phase 9.1)
        self_review = final_state.get("self_review")
        if self_review:
            out.subheader("Self-Review Assessment", "🔎")
            out.stat("Overall", f"{self_review.get('overall_score', 0):.2f}")
            out.stat("Relevance", f"{self_review.get('relevance_score', 0):.2f}")
            out.stat("Coverage", f"{self_review.get('coverage_score', 0):.2f}")
            out.stat("Coherence", f"{self_review.get('coherence_score', 0):.2f}")

            irrelevant = self_review.get("irrelevant_papers", [])
            if irrelevant:
                out.subheader(f"Irrelevant Papers ({len(irrelevant)})", "⚠")
                for p in irrelevant:
                    out.bullet(f"[{p.get('paper_id', '?')}] {p.get('title', '?')}: {p.get('reason', '')}")

            gaps = self_review.get("coverage_gaps", [])
            if gaps:
                out.subheader("Coverage Gaps", "📋")
                for gap in gaps[:3]:
                    out.bullet(gap)

            suggestions = self_review.get("suggestions", [])
            if suggestions:
                out.subheader("Suggestions", "💡")
                for s in suggestions[:3]:
                    out.bullet(s)

            if self_review.get("overall_score", 1.0) < 0.7:
                out.warning("Overall score below 0.7 — review may need revision")

        # Display citation grounding results (Phase 9.5)
        grounding = final_state.get("_citation_grounding")
        if grounding:
            rate = grounding.get("grounding_rate", 1.0)
            total = grounding.get("total_citations", 0)
            grounded = grounding.get("grounded_count", 0)
            ungrounded_count = grounding.get("ungrounded_count", 0)
            out.subheader("Citation Grounding", "🔗")
            out.stat("Citations Found", total)
            out.stat("Grounded", f"{grounded}/{total} ({rate:.0%})")
            if ungrounded_count > 0:
                out.warning(f"{ungrounded_count} ungrounded citation(s) — potential hallucinations")
                for cit in grounding.get("ungrounded", [])[:5]:
                    out.bullet(f"[{cit}]")

        if synthesis:
            # Always save output
            review_text = synthesis.get("review_text", "")

            # Save review markdown
            review_file = output_path.with_suffix(".md")
            with open(review_file, "w", encoding="utf-8") as f:
                f.write(f"# Literature Review: {research_question}\n\n")
                f.write(review_text)
                f.write("\n\n## References\n\n")
                for cit in synthesis.get("citations_formatted", []):
                    f.write(f"- {cit}\n")
            out.success(f"Review saved to: {review_file}", "📄")

            # Save full state as JSON
            json_file = output_path.with_suffix(".json")
            with open(json_file, "w", encoding="utf-8") as f:
                # Convert to JSON-serializable format
                output_data = {
                    "research_question": research_question,
                    "search_results": search_results,
                    "analyzed_papers": [dict(p) for p in analyzed],
                    "knowledge_graph": knowledge_graph,  # Phase 7.5
                    "synthesis": dict(synthesis) if synthesis else None,
                    "self_review": dict(self_review) if self_review else None,
                    "token_usage": final_state.get("_token_usage"),  # Phase 9.5
                    "citation_grounding": final_state.get("_citation_grounding"),  # Phase 9.5
                    "errors": errors,
                }
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            out.success(f"Full data saved to: {json_file}", "📊")

            # Show preview
            out.preview(
                "REVIEW PREVIEW",
                review_text,
                max_chars=500,
            )
            if len(review_text) > 500:
                out.info(f"[full review in {review_file}]")

            # Show token usage summary (Phase 9.5)
            token_usage = final_state.get("_token_usage")
            if token_usage:
                tracker = final_state.get("token_tracker")
                if tracker:
                    out.blank()
                    out.header("LLM COST SUMMARY")
                    for line in tracker.format_cli_summary().split("\n"):
                        out.info(line)

        else:
            out.error("No synthesis generated")

        # Post-review interactive menu (Emergency Fix Step 7b)
        session_id = final_state.get("_session_id")
        if session_id and synthesis:
            out.blank()
            out.stat("Session ID", session_id)
            while True:
                out.blank()
                out.info("What next?")
                out.bullet("[1] Save & exit (default)")
                out.bullet("[2] Refine review")
                out.bullet("[3] Show full review text")
                choice = input("  Choice (1/2/3): ").strip()
                if choice == "2":
                    instruction = input("  Refinement instruction: ").strip()
                    if instruction:
                        try:
                            out.info("Classifying instruction and refining review...")
                            result = await run_refinement(session_id, instruction)
                            out.success(f"Version {result['version_number']} created", "✅")
                            out.stat("Word Count", result["word_count"])
                            # Update saved files (include references)
                            review_file = output_path.with_suffix(".md")
                            citations = synthesis.get("citations_formatted", [])
                            with open(review_file, "w", encoding="utf-8") as f:
                                f.write(f"# Literature Review: {research_question}\n\n")
                                f.write(result["review_text"])
                                if citations:
                                    f.write("\n\n## References\n\n")
                                    for cit in citations:
                                        f.write(f"- {cit}\n")
                            # Update synthesis review_text for "Show full text"
                            synthesis["review_text"] = result["review_text"]
                            out.success(f"Updated: {review_file}", "📄")
                            out.preview("REFINED PREVIEW", result["review_text"], max_chars=500)
                            continue
                        except Exception as e:
                            out.error(f"Refinement failed: {e}")
                            continue
                    else:
                        out.info("No instruction provided, skipping.")
                        continue
                elif choice == "3":
                    review_text = synthesis.get("review_text", "")
                    out.blank()
                    print(review_text)
                    continue
                else:
                    break
        elif session_id:
            out.blank()
            out.stat("Session ID", session_id)
            out.info(f"Refine: litscribe session refine {session_id[:12]} -i \"your instruction\"")

        out.header("Review Complete")
        return 0

    except Exception as e:
        out.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


async def cmd_resume(args) -> int:
    """Resume an interrupted literature review from checkpoint."""
    from agents.graph import resume_literature_review

    log_file = setup_logging(verbose=args.verbose)
    out = get_output("litscribe.resume")

    thread_id = args.thread_id

    # Determine output path
    if getattr(args, "output", None):
        output_path = Path(args.output)
    else:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"review_resumed_{timestamp}"

    out.header("LitScribe Resume Review")
    out.stat("Thread ID", thread_id)
    out.stat("Output", output_path)
    out.stat("Log File", log_file)
    out.blank()
    out.info("Resuming from last checkpoint...")

    try:
        final_state = await resume_literature_review(
            thread_id=thread_id,
            verbose=args.verbose,
        )

        # Check for errors
        errors = final_state.get("errors", [])
        if errors:
            out.subheader("Warnings/Errors during processing", "⚠")
            for err in errors[-5:]:
                out.bullet(str(err))

        research_question = final_state.get("research_question", "Resumed review")

        # Display results
        analyzed = final_state.get("analyzed_papers", [])
        out.subheader("Papers Analyzed", "📖")
        out.stat("Count", len(analyzed))

        synthesis = final_state.get("synthesis")
        if synthesis:
            out.subheader("Review Generated", "✨")
            out.stat("Themes", len(synthesis.get("themes", [])))
            out.stat("Research Gaps", len(synthesis.get("gaps", [])))
            out.stat("Word Count", synthesis.get("word_count", 0))
            out.stat("Papers Cited", synthesis.get("papers_cited", 0))

            # Save review markdown
            review_text = synthesis.get("review_text", "")
            review_file = output_path.with_suffix(".md")
            with open(review_file, "w", encoding="utf-8") as f:
                f.write(f"# Literature Review: {research_question}\n\n")
                f.write(review_text)
                f.write("\n\n## References\n\n")
                for cit in synthesis.get("citations_formatted", []):
                    f.write(f"- {cit}\n")
            out.success(f"Review saved to: {review_file}", "📄")

            # Save full state as JSON
            json_file = output_path.with_suffix(".json")
            search_results = final_state.get("search_results")
            knowledge_graph = final_state.get("knowledge_graph")
            self_review = final_state.get("self_review")
            with open(json_file, "w", encoding="utf-8") as f:
                output_data = {
                    "research_question": research_question,
                    "search_results": search_results,
                    "analyzed_papers": [dict(p) for p in analyzed],
                    "knowledge_graph": knowledge_graph,
                    "synthesis": dict(synthesis) if synthesis else None,
                    "self_review": dict(self_review) if self_review else None,
                    "errors": errors,
                }
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            out.success(f"Full data saved to: {json_file}", "📊")

            # Show preview
            out.preview("REVIEW PREVIEW", review_text, max_chars=500)

        else:
            out.error("No synthesis generated")

        # Show session ID for refinement (Phase 9.3)
        session_id = final_state.get("_session_id")
        if session_id:
            out.blank()
            out.stat("Session ID", session_id)
            out.info(f"Refine: litscribe session refine {session_id[:12]} -i \"your instruction\"")

        out.header("Resume Complete")
        return 0

    except ValueError as e:
        out.error(f"Cannot resume: {e}")
        return 1
    except Exception as e:
        out.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


async def cmd_demo(args) -> int:
    """Run end-to-end demo pipeline."""
    from aggregators.unified_search import search_all_sources

    out = get_output("litscribe.demo")
    query = args.query or "large language model reasoning"

    out.header("LitScribe Demo Pipeline")
    out.stat("Query", query)
    out.blank()

    # Step 1: Multi-source search
    out.info("Step 1: Searching multiple sources...")
    try:
        result = await search_all_sources(
            query=query,
            sources=["arxiv", "semantic_scholar", "pubmed"],
            max_per_source=5,
            deduplicate=True,
            sort_by="citations",
        )

        out.stat("Found", f"{result['total_after_dedup']} unique papers")
        out.stat("Sources", result["source_counts"])

        papers = result["papers"]
        if not papers:
            out.warning("No papers found. Demo cannot continue.")
            return 1

        out.info("\n  Top 3 papers:")
        for i, paper in enumerate(papers[:3], 1):
            title = paper.get("title", "Unknown")[:50]
            cites = paper.get("citations", 0)
            out.stat(f"  {i}", f"{title}... ({cites} cites)")

    except Exception as e:
        out.error(f"Error in search: {e}")
        return 1

    # Step 2: Get detailed info for top paper
    out.info("\nStep 2: Fetching detailed metadata...")
    top_paper = papers[0]
    paper_id = None

    # Try to get a valid ID
    if top_paper.get("sources", {}).get("semantic_scholar"):
        paper_id = top_paper["sources"]["semantic_scholar"]
    elif top_paper.get("arxiv_id"):
        paper_id = f"arXiv:{top_paper['arxiv_id']}"
    elif top_paper.get("doi"):
        paper_id = top_paper["doi"]

    if paper_id:
        try:
            from mcp_servers.semantic_scholar_server import get_paper

            detail = await get_paper(paper_id)
            if "error" not in detail:
                out.stat("Title", f"{detail.get('title', 'Unknown')[:60]}...")
                out.stat("Citations", detail.get("citation_count", 0))
                out.stat("References", detail.get("reference_count", 0))
                if detail.get("tldr"):
                    out.stat("TL;DR", f"{detail['tldr'][:100]}...")
            else:
                out.warning(f"Could not fetch details: {detail['error']}")
        except Exception as e:
            out.error(f"Error fetching details: {e}")
    else:
        out.warning("No valid paper ID found for detailed fetch")

    # Step 3: Show deduplication in action
    out.info("\nStep 3: Deduplication analysis...")
    multi_source = [p for p in papers if len(p.get("sources", {})) > 1]
    if multi_source:
        out.success(f"Found {len(multi_source)} papers from multiple sources (merged)")
        for p in multi_source[:2]:
            out.bullet(f"{p['title'][:40]}... (sources: {list(p['sources'].keys())})")
    else:
        out.info("No duplicate papers found across sources")

    # Summary
    out.header("Demo Complete")
    out.info("LitScribe can:")
    out.bullet("Search arXiv, PubMed, Semantic Scholar in parallel")
    out.bullet("Deduplicate and merge papers from multiple sources")
    out.bullet("Fetch detailed metadata including citations and TL;DR")
    out.bullet("Parse PDFs to extract structured content")
    out.blank()

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="litscribe",
        description="LitScribe - Academic Literature Search and Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for papers
  litscribe search "transformer attention" --sources arxiv,semantic_scholar
  litscribe search "CRISPR" --sources pubmed --limit 20 --sort citations

  # Get paper details
  litscribe paper arXiv:1706.03762 --verbose
  litscribe citations arXiv:1706.03762 --limit 10

  # Parse PDF
  litscribe parse paper.pdf --output paper.md

  # Generate literature review (multi-agent system)
  litscribe review "What are the latest advances in LLM reasoning?"
  litscribe review "CRISPR applications" -s pubmed -p 15 -o my_review

  # Export review to different formats
  litscribe export review.json -f docx -s apa     # Export to Word (APA style)
  litscribe export review.json -f pdf -s ieee     # Export to PDF (IEEE style)
  litscribe export review.json -f bibtex          # Export BibTeX citations
  litscribe export review.json -f md -l zh        # Export Markdown in Chinese

  # Cache management
  litscribe cache stats              # Show cache statistics
  litscribe cache clear --expired    # Clear expired entries
  litscribe cache vacuum             # Optimize database

  # Run demo
  litscribe demo --query "multi-agent systems"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # search command
    search_parser = subparsers.add_parser("search", help="Search for papers across sources")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--sources", "-s",
        default="arxiv,semantic_scholar,pubmed",
        help="Comma-separated sources: arxiv,pubmed,semantic_scholar,zotero (default: arxiv,semantic_scholar,pubmed)",
    )
    search_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Max results per source (default: 10)",
    )
    search_parser.add_argument(
        "--sort",
        choices=["relevance", "citations", "year", "completeness"],
        default="relevance",
        help="Sort order (default: relevance)",
    )
    search_parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication",
    )
    search_parser.add_argument(
        "--output", "-o",
        help="Save results to JSON file",
    )
    search_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed paper info",
    )

    # paper command
    paper_parser = subparsers.add_parser("paper", help="Get detailed paper info")
    paper_parser.add_argument(
        "paper_id",
        help="Paper ID (DOI, arXiv:XXXX, PMID:XXXX, or S2 ID)",
    )
    paper_parser.add_argument(
        "--output", "-o",
        help="Save to JSON file",
    )
    paper_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show abstract, citations, references",
    )

    # citations command
    citations_parser = subparsers.add_parser("citations", help="Get papers citing a paper")
    citations_parser.add_argument(
        "paper_id",
        help="Paper ID",
    )
    citations_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Max citations to show (default: 20)",
    )

    # parse command
    parse_parser = subparsers.add_parser("parse", help="Parse a PDF file")
    parse_parser.add_argument("pdf_path", help="Path to PDF file")
    parse_parser.add_argument(
        "--output", "-o",
        help="Save markdown to file",
    )
    parse_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show section details",
    )

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo pipeline")
    demo_parser.add_argument(
        "--query", "-q",
        help="Search query for demo (default: 'large language model reasoning')",
    )

    # review command - Multi-agent literature review
    review_parser = subparsers.add_parser(
        "review",
        help="Generate a complete literature review using multi-agent system",
    )
    review_parser.add_argument(
        "question",
        help="Research question for the literature review",
    )
    review_parser.add_argument(
        "--papers", "-p",
        type=int,
        default=10,
        help="Maximum number of papers to analyze (default: 10, max: 500)",
    )
    review_parser.add_argument(
        "--sources", "-s",
        default="arxiv,semantic_scholar,pubmed",
        help="Comma-separated sources: arxiv,pubmed,semantic_scholar (default: arxiv,semantic_scholar,pubmed)",
    )
    review_parser.add_argument(
        "--type", "-t",
        choices=["narrative", "systematic", "scoping"],
        default="narrative",
        help="Type of literature review (default: narrative)",
    )
    review_parser.add_argument(
        "--output", "-o",
        help="Output file path (default: output/review_<question>_<timestamp>)",
    )
    review_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress and debug info",
    )
    # GraphRAG options (Phase 7.5)
    review_parser.add_argument(
        "--enable-graphrag",
        action="store_true",
        default=True,
        help="Enable GraphRAG knowledge graph (default: enabled)",
    )
    review_parser.add_argument(
        "--disable-graphrag",
        action="store_true",
        help="Disable GraphRAG for faster processing",
    )
    review_parser.add_argument(
        "--disable-self-review",
        action="store_true",
        help="Disable self-review agent (ablation)",
    )
    review_parser.add_argument(
        "--disable-domain-filter",
        action="store_true",
        help="Disable domain filtering in search (ablation)",
    )
    review_parser.add_argument(
        "--disable-snowball",
        action="store_true",
        help="Disable snowball sampling (ablation)",
    )
    review_parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for processing papers (default: 20)",
    )
    review_parser.add_argument(
        "--local-files", "-f",
        nargs="+",
        default=[],
        help="Local PDF files to include in review (paths)",
    )
    review_parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="en",
        help="Language for the generated review (default: en). 'zh' generates directly in Chinese.",
    )
    # Planning options (Phase 9.2)
    review_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only run the planning agent, show research plan, then stop",
    )

    # resume command - Resume interrupted review
    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume an interrupted literature review from checkpoint",
    )
    resume_parser.add_argument(
        "thread_id",
        help="Thread ID of the interrupted review (shown at start of 'litscribe review')",
    )
    resume_parser.add_argument(
        "--output", "-o",
        help="Output file path (default: output/review_resumed_<timestamp>)",
    )
    resume_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress and debug info",
    )

    # cache command - Cache management
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage the local cache",
    )
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command", help="Cache commands")

    # cache stats
    cache_stats_parser = cache_subparsers.add_parser("stats", help="Show cache statistics")

    # cache clear
    cache_clear_parser = cache_subparsers.add_parser("clear", help="Clear cache entries")
    cache_clear_parser.add_argument(
        "--expired",
        action="store_true",
        help="Only clear expired search cache entries",
    )
    cache_clear_parser.add_argument(
        "--search",
        action="store_true",
        help="Clear all search cache",
    )
    cache_clear_parser.add_argument(
        "--all",
        action="store_true",
        help="Clear all cache data (papers, PDFs, parsed docs)",
    )

    # cache vacuum
    cache_vacuum_parser = cache_subparsers.add_parser("vacuum", help="Optimize database storage")

    # session command - Review session management (Phase 9.3)
    session_parser = subparsers.add_parser(
        "session",
        help="Manage review sessions for iterative refinement",
    )
    session_subparsers = session_parser.add_subparsers(
        dest="session_command", help="Session commands"
    )

    # session list
    session_list_parser = session_subparsers.add_parser(
        "list", help="List all review sessions"
    )
    session_list_parser.add_argument(
        "--limit", "-l", type=int, default=20,
        help="Maximum sessions to show (default: 20)",
    )

    # session show <id>
    session_show_parser = session_subparsers.add_parser(
        "show", help="Show session details and version history"
    )
    session_show_parser.add_argument("session_id", help="Session ID (or prefix)")

    # session refine <id> -i "instruction"
    session_refine_parser = session_subparsers.add_parser(
        "refine", help="Refine a review with a natural language instruction"
    )
    session_refine_parser.add_argument("session_id", help="Session ID to refine")
    session_refine_parser.add_argument(
        "--instruction", "-i", required=True,
        help='Refinement instruction (e.g., "Add discussion about LoRA")',
    )
    session_refine_parser.add_argument(
        "--output", "-o", help="Save refined review to file",
    )

    # session diff <id> <v1> <v2>
    session_diff_parser = session_subparsers.add_parser(
        "diff", help="Show diff between two versions"
    )
    session_diff_parser.add_argument("session_id", help="Session ID")
    session_diff_parser.add_argument("v1", type=int, help="First version number")
    session_diff_parser.add_argument("v2", type=int, help="Second version number")

    # session rollback <id> <version>
    session_rollback_parser = session_subparsers.add_parser(
        "rollback", help="Rollback to a previous version"
    )
    session_rollback_parser.add_argument("session_id", help="Session ID")
    session_rollback_parser.add_argument("version", type=int, help="Version number to rollback to")

    # export command - Export review to various formats
    export_parser = subparsers.add_parser(
        "export",
        help="Export a saved review to various formats",
    )
    export_parser.add_argument(
        "input",
        help="Path to the review JSON file (from 'litscribe review' output)",
    )
    export_parser.add_argument(
        "--format", "-f",
        choices=["bibtex", "docx", "pdf", "html", "latex", "md"],
        default="docx",
        help="Output format (default: docx)",
    )
    export_parser.add_argument(
        "--style", "-s",
        choices=["apa", "mla", "ieee", "chicago", "gbt7714"],
        default="apa",
        help="Citation style (default: apa)",
    )
    export_parser.add_argument(
        "--lang", "-l",
        choices=["en", "zh"],
        default="en",
        help="Output language (default: en)",
    )
    export_parser.add_argument(
        "--output", "-o",
        help="Output file path (default: same as input with new extension)",
    )
    export_parser.add_argument(
        "--title",
        help="Custom title for the document",
    )
    export_parser.add_argument(
        "--author",
        help="Author name for the document",
    )

    return parser


async def cmd_cache(args) -> int:
    """Manage cache."""
    from cache import init_cache, get_cache_db

    out = get_output("litscribe.cache")

    try:
        db = init_cache()
    except Exception as e:
        out.error(f"Error initializing cache: {e}")
        return 1

    if args.cache_command == "stats":
        out.header("Cache Statistics")
        stats = db.get_stats()
        out.stat("Database", db.db_path)
        out.stat("Size", f"{stats.get('db_size_mb', 0)} MB")
        out.blank()
        out.info("Table counts:")
        out.stat("Papers cached", stats.get("papers_count", 0))
        out.stat("PDFs tracked", stats.get("pdfs_count", 0))
        out.stat("Parsed docs", stats.get("parsed_docs_count", 0))
        out.stat("Search queries", stats.get("search_cache_count", 0))
        out.stat("LLM responses", stats.get("llm_cache_count", 0))
        out.stat("Command logs", stats.get("command_logs_count", 0))
        # GraphRAG stats (Phase 7.5)
        out.stat("Entities", stats.get("entities_count", 0))
        out.stat("Entity mentions", stats.get("entity_mentions_count", 0))
        out.stat("Graph edges", stats.get("graph_edges_count", 0))
        out.stat("Communities", stats.get("communities_count", 0))
        out.blank()
        out.stat("Commands in last 7 days", stats.get("commands_last_7_days", 0))

    elif args.cache_command == "clear":
        if args.expired:
            count = db.clear_expired_cache()
            out.success(f"Cleared {count} expired cache entries")
        elif args.search:
            with db.get_connection() as conn:
                cursor = conn.execute("DELETE FROM search_cache")
                count = cursor.rowcount
                conn.commit()
            out.success(f"Cleared {count} search cache entries")
        elif args.all:
            confirm = input("This will delete all cached data. Are you sure? (y/N): ")
            if confirm.lower() == "y":
                with db.get_connection() as conn:
                    for table in ["search_cache", "llm_cache", "parsed_docs", "pdfs", "papers"]:
                        conn.execute(f"DELETE FROM {table}")
                    conn.commit()
                out.success("All cache data cleared")
            else:
                out.info("Aborted")
        else:
            out.warning("Specify what to clear: --expired, --search, or --all")
            return 1

    elif args.cache_command == "vacuum":
        out.info("Optimizing database...")
        db.vacuum()
        stats = db.get_stats()
        out.success(f"Done. Database size: {stats.get('db_size_mb', 0)} MB")

    else:
        out.warning("No cache command specified. Use: stats, clear, or vacuum")
        return 1

    return 0


async def cmd_session(args) -> int:
    """Manage review sessions for iterative refinement (Phase 9.3)."""
    out = get_output("litscribe.session")

    if not hasattr(args, "session_command") or args.session_command is None:
        out.warning("No session command specified. Use: list, show, refine, diff, rollback")
        return 1

    if args.session_command == "list":
        from versioning.review_versions import list_sessions

        sessions = list_sessions(limit=args.limit)
        if not sessions:
            out.info("No sessions found. Run 'litscribe review' to create one.")
            return 0

        out.header(f"Review Sessions ({len(sessions)})")
        for s in sessions:
            sid = s["session_id"]
            question = s.get("research_question", "?")[:60]
            created = s.get("created_at", "?")
            rtype = s.get("review_type", "?")
            out.stat(sid[:12] + "...", question)
            out.bullet(f"Created: {created}, Type: {rtype}, Lang: {s.get('language', '?')}")
        return 0

    elif args.session_command == "show":
        from versioning.review_versions import get_session, get_all_versions

        session = get_session(args.session_id)
        if not session:
            out.error(f"Session not found: {args.session_id}")
            return 1

        out.header("Session Details")
        out.stat("ID", session["session_id"])
        out.stat("Question", session["research_question"])
        out.stat("Type", session.get("review_type", "?"))
        out.stat("Language", session.get("language", "?"))
        out.stat("Created", session.get("created_at", "?"))
        out.stat("Updated", session.get("updated_at", "?"))

        versions = get_all_versions(session["session_id"])
        out.blank()
        out.subheader(f"Versions ({len(versions)})", "📋")
        for v in versions:
            instr = v.get("instruction") or "(original)"
            wc = v.get("word_count", "?")
            out.stat(f"v{v['version_number']}", f"{wc} words — {instr[:60]}")
        return 0

    elif args.session_command == "refine":
        from agents.graph import run_refinement

        out.header("Refining Review")
        out.stat("Session", args.session_id)
        out.stat("Instruction", args.instruction)
        out.blank()

        try:
            result = await run_refinement(
                session_id=args.session_id,
                instruction_text=args.instruction,
            )
            out.success(f"Version {result['version_number']} created", "✅")
            out.stat("Word Count", result["word_count"])
            out.stat("Action", result["instruction"].get("action_type", "unknown"))

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(result["review_text"])
                out.success(f"Saved to: {args.output}", "📄")
            else:
                out.preview("REFINED REVIEW", result["review_text"], max_chars=500)
            return 0
        except Exception as e:
            out.error(f"Refinement failed: {e}")
            return 1

    elif args.session_command == "diff":
        from versioning.review_versions import get_diff

        diff_text = get_diff(args.session_id, args.v1, args.v2)
        if not diff_text:
            out.info("No differences found (or versions do not exist).")
            return 0

        out.header(f"Diff: v{args.v1} → v{args.v2}")
        print(diff_text)
        return 0

    elif args.session_command == "rollback":
        from versioning.review_versions import rollback

        try:
            new_version = rollback(args.session_id, args.version)
            out.success(f"Rolled back to v{args.version}. New version: v{new_version}", "✅")
            return 0
        except Exception as e:
            out.error(f"Rollback failed: {e}")
            return 1

    else:
        out.warning("Unknown session command. Use: list, show, refine, diff, rollback")
        return 1


async def cmd_export(args) -> int:
    """Export a saved review to various formats."""
    from exporters.bibtex_exporter import BibTeXExporter
    from exporters.citation_formatter import CitationStyle
    from exporters.pandoc_exporter import ExportConfig, ExportFormat, PandocExporter

    out = get_output("litscribe.export")
    input_path = Path(args.input)
    if not input_path.exists():
        out.error(f"File not found: {input_path}")
        return 1

    if input_path.suffix != ".json":
        out.error("Input file must be a JSON file (from 'litscribe review' output)")
        return 1

    # Load the review data
    out.header("Export Literature Review")
    out.stat("Input", input_path)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        out.error(f"Invalid JSON file: {e}")
        return 1

    # Check for required data
    if not data.get("analyzed_papers"):
        out.error("No analyzed papers found in the review data")
        return 1

    # Map args to enums
    format_map = {
        "bibtex": None,  # Special handling
        "docx": ExportFormat.DOCX,
        "pdf": ExportFormat.PDF,
        "html": ExportFormat.HTML,
        "latex": ExportFormat.LATEX,
        "md": ExportFormat.MARKDOWN,
    }
    style_map = {
        "apa": CitationStyle.APA,
        "mla": CitationStyle.MLA,
        "ieee": CitationStyle.IEEE,
        "chicago": CitationStyle.CHICAGO,
        "gbt7714": CitationStyle.GB_T_7714,
    }

    output_format = args.format
    citation_style = style_map[args.style]
    language = args.lang

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        ext = ".bib" if output_format == "bibtex" else f".{output_format}"
        output_path = input_path.with_suffix(ext)

    out.stat("Format", output_format)
    out.stat("Citation style", args.style)
    out.stat("Language", language)
    out.stat("Output", output_path)
    out.blank()

    try:
        if output_format == "bibtex":
            # BibTeX export
            papers = data.get("analyzed_papers", [])
            exporter = BibTeXExporter(papers)
            output_path = exporter.save(output_path)
            out.success(f"BibTeX exported: {len(papers)} entries")
            out.stat("Cite keys", ", ".join(exporter.get_cite_keys()[:5]) + "...")

        else:
            # Pandoc export
            export_format = format_map[output_format]

            # Create state-like dict for the exporter
            state = {
                "research_question": data.get("research_question", "Literature Review"),
                "analyzed_papers": data.get("analyzed_papers", []),
                "synthesis": data.get("synthesis"),
            }

            config = ExportConfig(
                format=export_format,
                citation_style=citation_style,
                language=language,
                title=args.title,
                author=args.author,
            )

            exporter = PandocExporter(state, config)

            if output_format == "md" or not exporter._pandoc_available:
                if not exporter._pandoc_available and output_format != "md":
                    out.warning("Pandoc not installed. Exporting as Markdown instead.")
                    output_path = output_path.with_suffix(".md")

                output_path = exporter.export_markdown(output_path)
                out.success("Markdown exported successfully")
            else:
                output_path = exporter.export(output_path)
                out.success(f"{output_format.upper()} exported successfully")

        out.success(f"Saved to: {output_path}")
        return 0

    except Exception as e:
        out.error(f"Error during export: {e}")
        return 1


async def async_main(args) -> int:
    """Async main entry point."""
    if args.command == "search":
        return await cmd_search(args)
    elif args.command == "paper":
        return await cmd_paper(args)
    elif args.command == "citations":
        return await cmd_citations(args)
    elif args.command == "parse":
        return await cmd_parse(args)
    elif args.command == "demo":
        return await cmd_demo(args)
    elif args.command == "review":
        return await cmd_review(args)
    elif args.command == "resume":
        return await cmd_resume(args)
    elif args.command == "cache":
        return await cmd_cache(args)
    elif args.command == "session":
        return await cmd_session(args)
    elif args.command == "export":
        return await cmd_export(args)
    else:
        print("No command specified. Use --help for usage.")
        return 1


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())

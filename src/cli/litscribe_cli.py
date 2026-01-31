#!/usr/bin/env python3
"""LitScribe CLI - Command line interface for academic literature search and analysis."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_paper(paper: dict, index: int, verbose: bool = False) -> None:
    """Print a paper summary."""
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

    sources = args.sources.split(",") if args.sources else ["arxiv", "semantic_scholar"]

    print_header(f"Searching: '{args.query}'")
    print(f"Sources: {sources}")
    print(f"Max results per source: {args.limit}")
    print(f"Sort by: {args.sort}")
    print()

    try:
        result = await search_all_sources(
            query=args.query,
            sources=sources,
            max_per_source=args.limit,
            deduplicate=not args.no_dedup,
            sort_by=args.sort,
        )

        print(f"Results per source: {result['source_counts']}")
        print(f"Total before dedup: {result['total_before_dedup']}")
        print(f"Total after dedup: {result['total_after_dedup']}")
        print()

        papers = result["papers"]
        if not papers:
            print("No papers found.")
            return 0

        print_header("Results")
        for i, paper in enumerate(papers[: args.limit], 1):
            print_paper(paper, i, verbose=args.verbose)

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


async def cmd_parse(args) -> int:
    """Execute parse command."""
    from mcp_servers.pdf_parser_server import parse_pdf

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return 1

    print_header(f"Parsing: {pdf_path.name}")

    try:
        result = await parse_pdf(str(pdf_path))

        if "error" in result:
            print(f"Error: {result['error']}")
            return 1

        print(f"Title: {result.get('title', 'Unknown')}")
        print(f"Pages: {result.get('pages', 'N/A')}")
        print(f"Sections: {len(result.get('sections', []))}")
        print(f"Tables: {len(result.get('tables', []))}")
        print(f"References: {len(result.get('references', []))}")

        # Show sections
        if args.verbose and result.get("sections"):
            print("\nSections:")
            for sec in result["sections"][:10]:
                print(f"  - {sec.get('title', 'Untitled')} (p.{sec.get('start_page', '?')})")

        # Save markdown output
        if args.output:
            output_path = Path(args.output)
            content = result.get("markdown", "")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"\nMarkdown saved to: {output_path}")
        else:
            # Print first 500 chars of markdown
            markdown = result.get("markdown", "")
            if markdown:
                print("\n--- Preview (first 500 chars) ---")
                print(markdown[:500])
                if len(markdown) > 500:
                    print("...")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


async def cmd_paper(args) -> int:
    """Get detailed information about a specific paper."""
    from mcp_servers.semantic_scholar_server import get_paper

    print_header(f"Fetching: {args.paper_id}")

    try:
        result = await get_paper(args.paper_id)

        if "error" in result:
            print(f"Error: {result['error']}")
            return 1

        print(f"Title: {result.get('title', 'Unknown')}")
        print(f"Authors: {', '.join(result.get('authors', [])[:5])}")
        print(f"Year: {result.get('year', 'N/A')}")
        print(f"Venue: {result.get('venue', 'N/A')}")
        print(f"Citations: {result.get('citation_count', 0)}")
        print(f"References: {result.get('reference_count', 0)}")

        if result.get("doi"):
            print(f"DOI: {result['doi']}")
        if result.get("arxiv_id"):
            print(f"arXiv: {result['arxiv_id']}")
        if result.get("pdf_url"):
            print(f"PDF: {result['pdf_url']}")

        if result.get("tldr"):
            print(f"\nTL;DR: {result['tldr']}")

        if args.verbose:
            if result.get("abstract"):
                print(f"\nAbstract:\n{result['abstract']}")

            if result.get("top_citations"):
                print(f"\nTop Citations ({len(result['top_citations'])}):")
                for c in result["top_citations"][:5]:
                    print(f"  - {c.get('title', 'Untitled')[:60]}...")

            if result.get("top_references"):
                print(f"\nTop References ({len(result['top_references'])}):")
                for r in result["top_references"][:5]:
                    print(f"  - {r.get('title', 'Untitled')[:60]}...")

        # Save to file
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to: {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


async def cmd_citations(args) -> int:
    """Get citations for a paper."""
    from mcp_servers.semantic_scholar_server import get_paper_citations

    print_header(f"Citations for: {args.paper_id}")

    try:
        result = await get_paper_citations(args.paper_id, limit=args.limit)

        if "error" in result:
            print(f"Error: {result['error']}")
            return 1

        print(f"Total citations: {result.get('total', 0)}")
        print(f"Showing: {result.get('count', 0)}")
        print()

        for i, paper in enumerate(result.get("citations", []), 1):
            title = paper.get("title", "Unknown")[:60]
            year = paper.get("year", "N/A")
            cites = paper.get("citation_count", 0)
            print(f"{i}. [{year}] {title}... ({cites} cites)")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


async def cmd_review(args) -> int:
    """Run complete literature review using multi-agent system."""
    import warnings
    from datetime import datetime
    from agents.graph import run_literature_review

    # Suppress Pydantic serialization warnings from LiteLLM
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    research_question = args.question
    max_papers = args.papers
    sources = args.sources.split(",") if args.sources else ["arxiv", "semantic_scholar"]
    review_type = args.type

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

    print_header("LitScribe Literature Review")
    print(f"Research Question: {research_question}")
    print(f"Max Papers: {max_papers}")
    print(f"Sources: {sources}")
    print(f"Review Type: {review_type}")
    print(f"Output: {output_path}")
    print()
    print("Starting multi-agent workflow...")
    print("  1. Discovery Agent: Searching and selecting papers")
    print("  2. Critical Reading Agent: Analyzing papers")
    print("  3. Synthesis Agent: Generating review")
    print()

    try:
        # Run the multi-agent workflow
        final_state = await run_literature_review(
            research_question=research_question,
            max_papers=max_papers,
            sources=sources,
            review_type=review_type,
            verbose=args.verbose,
        )

        # Check for errors
        errors = final_state.get("errors", [])
        if errors:
            print(f"\n⚠ Warnings/Errors during processing:")
            for err in errors[-5:]:
                print(f"  - {err}")

        # Display results
        search_results = final_state.get("search_results")
        if search_results:
            print(f"\n📚 Papers Found: {search_results.get('total_found', 0)}")
            print(f"   Expanded Queries: {len(search_results.get('expanded_queries', []))}")
            print(f"   Source Distribution: {search_results.get('source_counts', {})}")

        analyzed = final_state.get("analyzed_papers", [])
        print(f"\n📖 Papers Analyzed: {len(analyzed)}")

        synthesis = final_state.get("synthesis")
        if synthesis:
            print(f"\n✨ Review Generated:")
            print(f"   Themes: {len(synthesis.get('themes', []))}")
            print(f"   Research Gaps: {len(synthesis.get('gaps', []))}")
            print(f"   Word Count: {synthesis.get('word_count', 0)}")
            print(f"   Papers Cited: {synthesis.get('papers_cited', 0)}")

            # Show themes
            if synthesis.get("themes"):
                print("\n📊 Identified Themes:")
                for i, theme in enumerate(synthesis["themes"], 1):
                    print(f"   {i}. {theme.get('theme', 'Unknown')}")
                    if args.verbose:
                        print(f"      {theme.get('description', '')[:100]}...")

            # Show gaps
            if synthesis.get("gaps"):
                print("\n🔍 Research Gaps:")
                for gap in synthesis["gaps"][:3]:
                    print(f"   - {gap[:80]}...")

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
            print(f"\n📄 Review saved to: {review_file}")

            # Save full state as JSON
            json_file = output_path.with_suffix(".json")
            with open(json_file, "w", encoding="utf-8") as f:
                # Convert to JSON-serializable format
                output_data = {
                    "research_question": research_question,
                    "search_results": search_results,
                    "analyzed_papers": [dict(p) for p in analyzed],
                    "synthesis": dict(synthesis) if synthesis else None,
                    "errors": errors,
                }
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"📊 Full data saved to: {json_file}")

            # Show preview
            print("\n" + "="*60)
            print("REVIEW PREVIEW (first 500 chars)")
            print("="*60)
            print(review_text[:500])
            if len(review_text) > 500:
                print(f"\n... [full review in {review_file}]")

        else:
            print("\n❌ No synthesis generated")

        print_header("Review Complete")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


async def cmd_demo(args) -> int:
    """Run end-to-end demo pipeline."""
    from aggregators.unified_search import search_all_sources

    query = args.query or "large language model reasoning"

    print_header("LitScribe Demo Pipeline")
    print(f"Query: '{query}'")
    print()

    # Step 1: Multi-source search
    print("Step 1: Searching multiple sources...")
    try:
        result = await search_all_sources(
            query=query,
            sources=["arxiv", "semantic_scholar"],
            max_per_source=5,
            deduplicate=True,
            sort_by="citations",
        )

        print(f"  Found {result['total_after_dedup']} unique papers")
        print(f"  Sources: {result['source_counts']}")

        papers = result["papers"]
        if not papers:
            print("  No papers found. Demo cannot continue.")
            return 1

        print("\n  Top 3 papers:")
        for i, paper in enumerate(papers[:3], 1):
            title = paper.get("title", "Unknown")[:50]
            cites = paper.get("citations", 0)
            print(f"    {i}. {title}... ({cites} cites)")

    except Exception as e:
        print(f"  Error in search: {e}")
        return 1

    # Step 2: Get detailed info for top paper
    print("\nStep 2: Fetching detailed metadata...")
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
                print(f"  Title: {detail.get('title', 'Unknown')[:60]}...")
                print(f"  Citations: {detail.get('citation_count', 0)}")
                print(f"  References: {detail.get('reference_count', 0)}")
                if detail.get("tldr"):
                    print(f"  TL;DR: {detail['tldr'][:100]}...")
            else:
                print(f"  Could not fetch details: {detail['error']}")
        except Exception as e:
            print(f"  Error fetching details: {e}")
    else:
        print("  No valid paper ID found for detailed fetch")

    # Step 3: Show deduplication in action
    print("\nStep 3: Deduplication analysis...")
    multi_source = [p for p in papers if len(p.get("sources", {})) > 1]
    if multi_source:
        print(f"  Found {len(multi_source)} papers from multiple sources (merged)")
        for p in multi_source[:2]:
            print(f"    - {p['title'][:40]}... (sources: {list(p['sources'].keys())})")
    else:
        print("  No duplicate papers found across sources")

    # Summary
    print_header("Demo Complete")
    print("LitScribe can:")
    print("  - Search arXiv, PubMed, Semantic Scholar in parallel")
    print("  - Deduplicate and merge papers from multiple sources")
    print("  - Fetch detailed metadata including citations and TL;DR")
    print("  - Parse PDFs to extract structured content")
    print()

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
        default="arxiv,semantic_scholar",
        help="Comma-separated sources: arxiv,pubmed,semantic_scholar,zotero (default: arxiv,semantic_scholar)",
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
        help="Maximum number of papers to analyze (default: 10)",
    )
    review_parser.add_argument(
        "--sources", "-s",
        default="arxiv,semantic_scholar",
        help="Comma-separated sources: arxiv,pubmed,semantic_scholar (default: arxiv,semantic_scholar)",
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

    try:
        db = init_cache()
    except Exception as e:
        print(f"Error initializing cache: {e}")
        return 1

    if args.cache_command == "stats":
        print_header("Cache Statistics")
        stats = db.get_stats()
        print(f"Database: {db.db_path}")
        print(f"Size: {stats.get('db_size_mb', 0)} MB")
        print()
        print("Table counts:")
        print(f"  Papers cached: {stats.get('papers_count', 0)}")
        print(f"  PDFs tracked: {stats.get('pdfs_count', 0)}")
        print(f"  Parsed docs: {stats.get('parsed_docs_count', 0)}")
        print(f"  Search queries: {stats.get('search_cache_count', 0)}")
        print(f"  LLM responses: {stats.get('llm_cache_count', 0)}")
        print(f"  Command logs: {stats.get('command_logs_count', 0)}")
        print()
        print(f"Commands in last 7 days: {stats.get('commands_last_7_days', 0)}")

    elif args.cache_command == "clear":
        if args.expired:
            count = db.clear_expired_cache()
            print(f"Cleared {count} expired cache entries")
        elif args.search:
            with db.get_connection() as conn:
                cursor = conn.execute("DELETE FROM search_cache")
                count = cursor.rowcount
                conn.commit()
            print(f"Cleared {count} search cache entries")
        elif args.all:
            confirm = input("This will delete all cached data. Are you sure? (y/N): ")
            if confirm.lower() == 'y':
                with db.get_connection() as conn:
                    for table in ["search_cache", "llm_cache", "parsed_docs", "pdfs", "papers"]:
                        conn.execute(f"DELETE FROM {table}")
                    conn.commit()
                print("All cache data cleared")
            else:
                print("Aborted")
        else:
            print("Specify what to clear: --expired, --search, or --all")
            return 1

    elif args.cache_command == "vacuum":
        print("Optimizing database...")
        db.vacuum()
        stats = db.get_stats()
        print(f"Done. Database size: {stats.get('db_size_mb', 0)} MB")

    else:
        print("No cache command specified. Use: stats, clear, or vacuum")
        return 1

    return 0


async def cmd_export(args) -> int:
    """Export a saved review to various formats."""
    from exporters.bibtex_exporter import BibTeXExporter
    from exporters.citation_formatter import CitationStyle
    from exporters.pandoc_exporter import ExportConfig, ExportFormat, PandocExporter

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    if input_path.suffix != ".json":
        print(f"Error: Input file must be a JSON file (from 'litscribe review' output)")
        return 1

    # Load the review data
    print_header("Export Literature Review")
    print(f"Input: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        return 1

    # Check for required data
    if not data.get("analyzed_papers"):
        print("Error: No analyzed papers found in the review data")
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

    print(f"Format: {output_format}")
    print(f"Citation style: {args.style}")
    print(f"Language: {language}")
    print(f"Output: {output_path}")
    print()

    try:
        if output_format == "bibtex":
            # BibTeX export
            papers = data.get("analyzed_papers", [])
            exporter = BibTeXExporter(papers)
            output_path = exporter.save(output_path)
            print(f"BibTeX exported: {len(papers)} entries")
            print(f"Cite keys: {', '.join(exporter.get_cite_keys()[:5])}...")

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
                    print(f"Warning: Pandoc not installed. Exporting as Markdown instead.")
                    output_path = output_path.with_suffix(".md")

                output_path = exporter.export_markdown(output_path)
                print(f"Markdown exported successfully")
            else:
                output_path = exporter.export(output_path)
                print(f"{output_format.upper()} exported successfully")

        print(f"\nSaved to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error during export: {e}")
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
    elif args.command == "cache":
        return await cmd_cache(args)
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

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
  litscribe search "transformer attention" --sources arxiv,semantic_scholar
  litscribe search "CRISPR" --sources pubmed --limit 20 --sort citations
  litscribe paper arXiv:1706.03762 --verbose
  litscribe citations arXiv:1706.03762 --limit 10
  litscribe parse paper.pdf --output paper.md
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

    return parser


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

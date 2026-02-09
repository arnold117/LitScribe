#!/usr/bin/env python3
"""
LitScribe Demo Workflow

Demonstrates the complete pipeline:
1. Multi-source paper search (arXiv + Semantic Scholar + PubMed)
2. Paper deduplication and ranking
3. Detailed metadata retrieval
4. PDF parsing (if available)
5. Zotero integration (if configured)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_step(step_num: int, title: str) -> None:
    print(f"\n[Step {step_num}] {title}")
    print("-" * 50)


async def demo_multi_source_search():
    """Demonstrate multi-source search with deduplication."""
    from aggregators.unified_search import search_all_sources

    print_header("LitScribe MVP Demo - Multi-Source Academic Search")

    query = "attention mechanism transformer"

    print(f"Query: '{query}'")
    print(f"This demo will search arXiv and Semantic Scholar simultaneously,")
    print(f"deduplicate results, and show unified paper information.\n")

    # Step 1: Search multiple sources
    print_step(1, "Searching Multiple Sources in Parallel")

    result = await search_all_sources(
        query=query,
        sources=["arxiv", "semantic_scholar"],
        max_per_source=10,
        deduplicate=True,
        sort_by="citations",
    )

    print(f"Sources searched: {result['sources_searched']}")
    print(f"Results per source: {result['source_counts']}")
    print(f"Total before deduplication: {result['total_before_dedup']}")
    print(f"Total after deduplication: {result['total_after_dedup']}")

    papers = result["papers"]
    if not papers:
        print("\nNo papers found. Check network connection or try again later.")
        return

    # Step 2: Show top results
    print_step(2, "Top 5 Papers (Sorted by Citations)")

    for i, paper in enumerate(papers[:5], 1):
        title = paper.get("title", "Unknown")
        if len(title) > 60:
            title = title[:57] + "..."

        authors = paper.get("authors", [])
        author_str = ", ".join(authors[:2])
        if len(authors) > 2:
            author_str += f" (+{len(authors)-2} more)"

        print(f"\n{i}. {title}")
        print(f"   Authors: {author_str}")
        print(f"   Year: {paper.get('year', 'N/A')}")
        print(f"   Citations: {paper.get('citations', 0)}")
        print(f"   Sources: {list(paper.get('sources', {}).keys())}")

        if paper.get("doi"):
            print(f"   DOI: {paper['doi']}")
        if paper.get("arxiv_id"):
            print(f"   arXiv: {paper['arxiv_id']}")

    # Step 3: Analyze deduplication
    print_step(3, "Deduplication Analysis")

    multi_source_papers = [p for p in papers if len(p.get("sources", {})) > 1]
    if multi_source_papers:
        print(f"Found {len(multi_source_papers)} papers from multiple sources:")
        for p in multi_source_papers[:3]:
            print(f"  - {p['title'][:50]}...")
            print(f"    Sources: {list(p['sources'].keys())}")
    else:
        print("No duplicate papers found across sources in this result set.")
        print("(This is common when sources have different coverage)")

    # Step 4: Get detailed info for top paper
    print_step(4, "Fetching Detailed Metadata for Top Paper")

    top_paper = papers[0]
    paper_id = None

    # Try different ID types
    if top_paper.get("sources", {}).get("semantic_scholar"):
        paper_id = top_paper["sources"]["semantic_scholar"]
    elif top_paper.get("arxiv_id"):
        paper_id = f"arXiv:{top_paper['arxiv_id']}"
    elif top_paper.get("doi"):
        paper_id = top_paper["doi"]

    if paper_id:
        try:
            from services.semantic_scholar import get_paper

            print(f"Fetching details for: {paper_id}")
            detail = await get_paper(paper_id)

            if "error" not in detail:
                print(f"\nTitle: {detail.get('title', 'Unknown')}")
                print(f"Citation Count: {detail.get('citation_count', 0)}")
                print(f"Reference Count: {detail.get('reference_count', 0)}")
                print(f"Venue: {detail.get('venue', 'N/A')}")

                if detail.get("tldr"):
                    print(f"\nTL;DR (AI Summary):")
                    print(f"  {detail['tldr']}")

                if detail.get("top_citations"):
                    print(f"\nTop Citing Papers:")
                    for c in detail["top_citations"][:3]:
                        print(f"  - {c.get('title', 'Unknown')[:50]}...")
            else:
                print(f"Could not fetch details: {detail['error']}")
                print("(This may be due to rate limiting - try again in a few minutes)")
        except Exception as e:
            print(f"Error fetching details: {e}")
    else:
        print("No valid paper ID available for detailed fetch")

    # Summary
    print_header("Demo Complete")
    print("LitScribe MVP successfully demonstrated:")
    print("  1. Parallel multi-source search (arXiv + Semantic Scholar)")
    print("  2. Automatic paper deduplication based on DOI/title similarity")
    print("  3. Unified paper model with normalized metadata")
    print("  4. Ranking by citations, relevance, year, or completeness")
    print("  5. Detailed paper info including TL;DR summaries")
    print()
    print("Additional capabilities (not shown in this demo):")
    print("  - PubMed search for biomedical literature")
    print("  - Zotero integration for local library management")
    print("  - PDF parsing to extract structured content")
    print()


async def demo_pubmed_search():
    """Demonstrate PubMed search for biomedical topics."""
    from aggregators.unified_search import search_all_sources

    print_header("PubMed Search Demo - Biomedical Literature")

    query = "CRISPR gene editing therapy"
    print(f"Query: '{query}'\n")

    result = await search_all_sources(
        query=query,
        sources=["pubmed"],
        max_per_source=5,
        sort_by="year",
    )

    print(f"Found {result['total_after_dedup']} papers from PubMed\n")

    for i, paper in enumerate(result["papers"][:5], 1):
        title = paper.get("title", "Unknown")[:60]
        print(f"{i}. [{paper.get('year', 'N/A')}] {title}...")
        if paper.get("pmid"):
            print(f"   PMID: {paper['pmid']}")
        if paper.get("mesh_terms"):
            print(f"   MeSH: {', '.join(paper['mesh_terms'][:3])}")


async def demo_pdf_parsing():
    """Demonstrate PDF parsing (requires a sample PDF)."""
    print_header("PDF Parsing Demo")

    # Check for sample PDFs in data directory
    data_dir = Path(__file__).parent.parent / "data" / "pdfs"

    if not data_dir.exists():
        print(f"PDF directory not found: {data_dir}")
        print("To test PDF parsing, place a PDF file in data/pdfs/ and run:")
        print("  python src/cli/litscribe_cli.py parse <path-to-pdf>")
        return

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/pdfs/")
        print("To test PDF parsing, place a PDF file there and run:")
        print("  python src/cli/litscribe_cli.py parse <path-to-pdf>")
        return

    # Parse first PDF found
    pdf_path = pdf_files[0]
    print(f"Parsing: {pdf_path.name}\n")

    try:
        from services.pdf_parser import parse_pdf

        result = await parse_pdf(str(pdf_path))

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print(f"Title: {result.get('title', 'Unknown')}")
        print(f"Pages: {result.get('pages', 'N/A')}")
        print(f"Sections: {len(result.get('sections', []))}")
        print(f"Tables: {len(result.get('tables', []))}")
        print(f"References: {len(result.get('references', []))}")

        if result.get("sections"):
            print("\nExtracted Sections:")
            for sec in result["sections"][:5]:
                print(f"  - {sec.get('title', 'Untitled')}")

    except Exception as e:
        print(f"Error parsing PDF: {e}")


async def main():
    """Run all demos."""
    import argparse

    parser = argparse.ArgumentParser(description="LitScribe Demo Workflow")
    parser.add_argument(
        "--demo",
        choices=["all", "search", "pubmed", "pdf"],
        default="search",
        help="Which demo to run (default: search)",
    )
    args = parser.parse_args()

    if args.demo == "all":
        await demo_multi_source_search()
        print("\n" + "=" * 70 + "\n")
        await demo_pubmed_search()
        print("\n" + "=" * 70 + "\n")
        await demo_pdf_parsing()
    elif args.demo == "search":
        await demo_multi_source_search()
    elif args.demo == "pubmed":
        await demo_pubmed_search()
    elif args.demo == "pdf":
        await demo_pdf_parsing()


if __name__ == "__main__":
    asyncio.run(main())

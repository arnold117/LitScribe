#!/usr/bin/env python3
"""Test script for Semantic Scholar MCP Server."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_servers.semantic_scholar_server import (
    search_papers,
    get_paper,
    get_paper_citations,
    get_paper_references,
    search_by_author,
    get_recommendations,
    batch_get_papers,
)


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def test_search():
    """Test paper search."""
    print_header("1. Paper Search")

    result = await search_papers(
        query="transformer attention mechanism",
        limit=5,
    )

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return None

    print(f"Query: 'transformer attention mechanism'")
    print(f"Total results: {result['total']}")
    print(f"Returned: {result['count']}")

    first_paper_id = None
    for i, paper in enumerate(result["papers"], 1):
        print(f"\n{i}. {paper['title'][:60]}...")
        print(f"   Authors: {', '.join(paper['authors'][:3])}" + (" ..." if len(paper['authors']) > 3 else ""))
        print(f"   Year: {paper['year']}, Citations: {paper['citation_count']}")
        if paper.get("tldr"):
            print(f"   TLDR: {paper['tldr'][:80]}...")
        if paper.get("pdf_url"):
            print(f"   PDF: {paper['pdf_url'][:50]}...")
        if i == 1:
            first_paper_id = paper["paper_id"]

    print("\n✅ Search works!")
    return first_paper_id


async def test_get_paper():
    """Test getting paper by various IDs."""
    print_header("2. Get Paper by ID")

    # Test with arXiv ID (Attention Is All You Need)
    paper_id = "arXiv:1706.03762"
    print(f"Fetching paper: {paper_id}")

    result = await get_paper(paper_id)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"\nTitle: {result['title']}")
    print(f"Authors: {', '.join(result['authors'][:5])}")
    print(f"Year: {result['year']}")
    print(f"Citations: {result['citation_count']}")
    print(f"References: {result['reference_count']}")
    print(f"Venue: {result['venue']}")
    print(f"DOI: {result.get('doi')}")
    print(f"arXiv: {result.get('arxiv_id')}")

    if result.get("top_citations"):
        print(f"\nTop citations ({len(result['top_citations'])}):")
        for c in result["top_citations"][:3]:
            print(f"  - {c['title'][:50]}...")

    if result.get("top_references"):
        print(f"\nTop references ({len(result['top_references'])}):")
        for r in result["top_references"][:3]:
            print(f"  - {r['title'][:50]}...")

    print("\n✅ Get paper works!")


async def test_citations():
    """Test getting citations."""
    print_header("3. Paper Citations")

    paper_id = "arXiv:1706.03762"  # Attention paper
    print(f"Getting citations for: {paper_id}")

    result = await get_paper_citations(paper_id, limit=10)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"Total citations: {result['total']}")
    print(f"Returned: {result['count']}")

    for i, paper in enumerate(result["citations"][:5], 1):
        print(f"\n{i}. {paper['title'][:55]}...")
        print(f"   Year: {paper['year']}, Citations: {paper['citation_count']}")

    print("\n✅ Citations work!")


async def test_references():
    """Test getting references."""
    print_header("4. Paper References")

    paper_id = "arXiv:1706.03762"
    print(f"Getting references for: {paper_id}")

    result = await get_paper_references(paper_id, limit=10)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"Total references: {result['total']}")
    print(f"Returned: {result['count']}")

    for i, paper in enumerate(result["references"][:5], 1):
        print(f"\n{i}. {paper['title'][:55]}...")
        print(f"   Year: {paper['year']}, Citations: {paper['citation_count']}")

    print("\n✅ References work!")


async def test_author_search():
    """Test author search."""
    print_header("5. Author Search")

    result = await search_by_author("Yoshua Bengio", limit=5)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"Found {result['count']} authors")

    for author in result["authors"][:2]:
        print(f"\nAuthor: {author['name']}")
        print(f"  Papers: {author['paper_count']}, Citations: {author['citation_count']}")
        print(f"  h-index: {author['h_index']}")
        print(f"  Affiliations: {author['affiliations']}")

        if author["papers"]:
            print(f"  Recent papers:")
            for p in author["papers"][:3]:
                print(f"    - {p['title'][:45]}... ({p['year']}, {p['citation_count']} cites)")

    print("\n✅ Author search works!")


async def test_recommendations():
    """Test paper recommendations."""
    print_header("6. Paper Recommendations")

    paper_id = "arXiv:1706.03762"
    print(f"Getting recommendations based on: {paper_id}")

    result = await get_recommendations(paper_id, limit=5)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"Found {result['count']} recommendations")

    for i, paper in enumerate(result["recommendations"], 1):
        print(f"\n{i}. {paper['title'][:55]}...")
        print(f"   Year: {paper['year']}, Citations: {paper['citation_count']}")

    print("\n✅ Recommendations work!")


async def test_batch():
    """Test batch paper retrieval."""
    print_header("7. Batch Get Papers")

    paper_ids = [
        "arXiv:1706.03762",  # Attention Is All You Need
        "arXiv:1810.04805",  # BERT
        "arXiv:2005.14165",  # GPT-3
    ]

    print(f"Fetching {len(paper_ids)} papers...")

    result = await batch_get_papers(paper_ids)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"Requested: {result['requested']}, Found: {result['found']}")

    for i, paper in enumerate(result["papers"], 1):
        print(f"\n{i}. {paper['title'][:50]}...")
        print(f"   Year: {paper['year']}, Citations: {paper['citation_count']}")

    print("\n✅ Batch get works!")


async def test_year_filter():
    """Test search with year filter."""
    print_header("8. Search with Year Filter")

    result = await search_papers(
        query="large language model",
        limit=5,
        year="2023-2024",
    )

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"Query: 'large language model' (2023-2024)")
    print(f"Found: {result['count']} papers")

    for i, paper in enumerate(result["papers"], 1):
        print(f"  {i}. [{paper['year']}] {paper['title'][:45]}... ({paper['citation_count']} cites)")

    print("\n✅ Year filter works!")


async def main():
    print("\n" + "Semantic Scholar MCP Server Test".center(60))
    print("=" * 60)

    await test_search()
    await test_get_paper()
    await test_citations()
    await test_references()
    await test_author_search()
    await test_recommendations()
    await test_batch()
    await test_year_filter()

    print_header("Summary")
    print("✅ All Semantic Scholar API tests completed!")
    print("\nFeatures available:")
    print("  - Paper search with filters (year, field, open access)")
    print("  - Get paper by DOI, arXiv ID, or S2 ID")
    print("  - Citation and reference tracking")
    print("  - Author search with h-index")
    print("  - AI-generated TLDR summaries")
    print("  - Paper recommendations")
    print("  - Batch paper retrieval")


if __name__ == "__main__":
    asyncio.run(main())

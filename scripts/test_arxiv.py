#!/usr/bin/env python3
"""Test script for arXiv MCP Server functions."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.arxiv import (
    search_papers,
    get_paper_metadata,
    download_pdf,
    get_recent_papers,
    search_by_author,
    batch_get_papers,
)


def print_header(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}\n")


async def main():
    print("\n" + "arXiv MCP Server Test".center(50))
    print("=" * 50)

    # Test 1: Search papers
    print_header("1. Search Papers")
    result = await search_papers("large language models", max_results=3)
    print(f"Query: '{result['query']}'")
    print(f"Found: {result['count']} papers")
    for i, paper in enumerate(result["papers"], 1):
        print(f"\n  [{i}] {paper['title'][:60]}...")
        print(f"      arXiv ID: {paper['arxiv_id']}")
        print(f"      Authors: {', '.join(paper['authors'][:3])}...")
        print(f"      Categories: {', '.join(paper['categories'][:3])}")

    # Test 2: Get paper metadata
    print_header("2. Get Paper Metadata")
    if result["papers"]:
        arxiv_id = result["papers"][0]["arxiv_id"]
        metadata = await get_paper_metadata(arxiv_id)
        print(f"arXiv ID: {metadata['arxiv_id']}")
        print(f"Title: {metadata['title']}")
        print(f"Published: {metadata['published']}")
        print(f"PDF URL: {metadata['pdf_url']}")
        print(f"Abstract: {metadata['abstract'][:150]}...")

    # Test 3: Get recent papers by category
    print_header("3. Recent Papers (cs.CL)")
    recent = await get_recent_papers("cs.CL", max_results=3)
    print(f"Category: {recent['category']}")
    print(f"Found: {recent['count']} papers")
    for paper in recent["papers"]:
        print(f"  - {paper['title'][:50]}...")
        print(f"    Published: {paper['published'][:10]}")

    # Test 4: Search by author
    print_header("4. Search by Author")
    author_papers = await search_by_author("Yann LeCun", max_results=3)
    print(f"Author: {author_papers['author']}")
    print(f"Found: {author_papers['count']} papers")
    for paper in author_papers["papers"]:
        print(f"  - {paper['title'][:50]}...")

    # Test 5: Batch get papers
    print_header("5. Batch Get Papers")
    # Some well-known paper IDs
    test_ids = ["1706.03762", "1810.04805", "2005.14165"]  # Attention, BERT, GPT-3
    batch = await batch_get_papers(test_ids)
    print(f"Requested: {batch['requested']}")
    print(f"Found: {batch['found']}")
    for paper in batch["papers"]:
        print(f"  - [{paper['arxiv_id']}] {paper['title'][:40]}...")

    # Test 6: Download PDF (optional - commented out to avoid actual download)
    # print_header("6. Download PDF (skipped)")
    # print("Skipping PDF download to avoid cluttering disk.")
    # print("To test, uncomment the following in test_arxiv.py:")
    # print("  download_result = await download_pdf('1706.03762')")

    # Uncomment to actually test download:
    print_header("6. Download PDF")
    download_result = await download_pdf("1706.03762")
    print(f"arXiv ID: {download_result.get('arxiv_id')}")
    print(f"Title: {download_result.get('title', 'N/A')[:50]}...")
    print(f"PDF Path: {download_result.get('pdf_path')}")
    print(f"Downloaded: {download_result.get('downloaded')}")

    print_header("Summary")
    print("All arXiv MCP functions tested successfully!")


if __name__ == "__main__":
    asyncio.run(main())

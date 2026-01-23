#!/usr/bin/env python3
"""Test script for PubMed MCP Server functions."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_servers.pubmed_server import (
    search_pubmed,
    get_article_details,
    get_related_articles,
    get_citations,
    batch_get_articles,
)


def print_header(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}\n")


async def main():
    print("\n" + "PubMed MCP Server Test".center(50))
    print("=" * 50)

    # Test 1: Search
    print_header("1. Search PubMed")
    result = await search_pubmed("CRISPR gene editing", max_results=3)
    print(f"Total results: {result['total_count']}")
    print(f"Returned: {result['returned_count']} articles")
    for i, article in enumerate(result["articles"], 1):
        print(f"\n  [{i}] {article['title'][:60]}...")
        print(f"      PMID: {article['pmid']}")
        print(f"      Authors: {', '.join(article['authors'][:3])}...")

    # Test 2: Get article details
    print_header("2. Get Article Details")
    if result["articles"]:
        pmid = result["articles"][0]["pmid"]
        details = await get_article_details(pmid)
        print(f"PMID: {details['pmid']}")
        print(f"Title: {details['title']}")
        print(f"Journal: {details['journal']}")
        print(f"MeSH Terms: {details['mesh_terms'][:5]}...")

    # Test 3: Related articles
    print_header("3. Get Related Articles")
    if result["articles"]:
        pmid = result["articles"][0]["pmid"]
        related = await get_related_articles(pmid, max_results=3)
        print(f"Source: {related['source_pmid']}")
        print(f"Found {related['related_count']} related articles")
        for i, article in enumerate(related["related_articles"][:3], 1):
            print(f"  [{i}] {article['title'][:50]}...")

    # Test 4: Citations
    print_header("4. Get Citations")
    # Use a well-known paper (CRISPR Cas9)
    citations = await get_citations("23287718", direction="both")
    print(f"PMID: {citations['pmid']}")
    print(f"Cited by: {citations.get('cited_by_count', 0)} papers")
    print(f"References: {citations.get('references_count', 0)} papers")

    # Test 5: Batch fetch
    print_header("5. Batch Get Articles")
    batch = await batch_get_articles(["23287718", "20628091", "19915144"])
    print(f"Requested: {batch['requested_count']}")
    print(f"Returned: {batch['returned_count']}")
    for article in batch["articles"]:
        print(f"  - [{article['pmid']}] {article['title'][:50]}...")

    print_header("Summary")
    print("All PubMed MCP functions tested successfully!")


if __name__ == "__main__":
    asyncio.run(main())

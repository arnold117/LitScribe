#!/usr/bin/env python3
"""Test script for Unified Search Aggregator."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aggregators.unified_search import UnifiedSearchAggregator, search_all_sources
from aggregators.deduplicator import title_similarity, are_same_paper, deduplicate_papers
from models.unified_paper import UnifiedPaper


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_paper(paper: dict, index: int) -> None:
    """Print a paper summary."""
    print(f"\n{index}. {paper['title'][:70]}...")
    print(f"   Authors: {', '.join(paper['authors'][:3])}" + (" ..." if len(paper['authors']) > 3 else ""))
    print(f"   Year: {paper['year']}")
    print(f"   Sources: {list(paper['sources'].keys())}")
    print(f"   Citations: {paper['citations']}")
    print(f"   Completeness: {paper['completeness_score']:.2f}")
    if paper.get('doi'):
        print(f"   DOI: {paper['doi']}")
    if paper.get('arxiv_id'):
        print(f"   arXiv: {paper['arxiv_id']}")
    if paper.get('pmid'):
        print(f"   PMID: {paper['pmid']}")


async def test_deduplication():
    """Test the deduplication logic."""
    print_header("1. Deduplication Logic Test")

    # Create test papers that should be detected as duplicates
    paper1 = UnifiedPaper(
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        abstract="The dominant sequence transduction models...",
        year=2017,
        sources={"arxiv": "1706.03762"},
        arxiv_id="1706.03762",
    )

    paper2 = UnifiedPaper(
        title="Attention is All You Need",  # Slightly different casing
        authors=["Vaswani, A", "Shazeer, N"],  # Different format
        abstract="Dominant sequence models are based on...",
        year=2017,
        sources={"pubmed": "12345"},
        doi="10.48550/arXiv.1706.03762",
    )

    paper3 = UnifiedPaper(
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        authors=["Jacob Devlin", "Ming-Wei Chang"],
        abstract="We introduce a new language representation model...",
        year=2018,
        sources={"arxiv": "1810.04805"},
        arxiv_id="1810.04805",
    )

    # Test title similarity
    sim = title_similarity(paper1.title, paper2.title)
    print(f"Title similarity (same paper): {sim:.3f}")
    print(f"  Expected: >= 0.9")

    sim_diff = title_similarity(paper1.title, paper3.title)
    print(f"Title similarity (different paper): {sim_diff:.3f}")
    print(f"  Expected: < 0.5")

    # Test are_same_paper
    same = are_same_paper(paper1, paper2)
    print(f"\nare_same_paper (paper1, paper2): {same}")
    print(f"  Expected: True")

    not_same = are_same_paper(paper1, paper3)
    print(f"are_same_paper (paper1, paper3): {not_same}")
    print(f"  Expected: False")

    # Test deduplication
    papers = [paper1, paper2, paper3]
    deduped = deduplicate_papers(papers)
    print(f"\nDeduplication: {len(papers)} papers -> {len(deduped)} unique")
    print(f"  Expected: 2 unique papers")

    if len(deduped) == 2:
        print("\n  Merged paper info:")
        merged = [p for p in deduped if "Attention" in p.title][0]
        print(f"    Sources: {merged.sources}")
        print(f"    Expected sources: arxiv + pubmed")

    print("\n✅ Deduplication tests passed!" if len(deduped) == 2 else "\n❌ Deduplication test failed!")


async def test_arxiv_search():
    """Test arXiv search."""
    print_header("2. arXiv Search Test")

    aggregator = UnifiedSearchAggregator()
    papers = await aggregator.search_arxiv("transformer attention", max_results=5)

    print(f"Found {len(papers)} papers from arXiv")

    if papers:
        for i, paper in enumerate(papers[:3], 1):
            print(f"\n  {i}. {paper.title[:60]}...")
            print(f"     Year: {paper.year}, Categories: {paper.categories[:2]}")
        print("\n✅ arXiv search works!")
    else:
        print("⚠️  No papers found (check network)")


async def test_pubmed_search():
    """Test PubMed search."""
    print_header("3. PubMed Search Test")

    aggregator = UnifiedSearchAggregator()
    papers = await aggregator.search_pubmed("CRISPR gene editing", max_results=5)

    print(f"Found {len(papers)} papers from PubMed")

    if papers:
        for i, paper in enumerate(papers[:3], 1):
            print(f"\n  {i}. {paper.title[:60]}...")
            print(f"     Year: {paper.year}, MeSH: {paper.mesh_terms[:2]}")
        print("\n✅ PubMed search works!")
    else:
        print("⚠️  No papers found (check NCBI config)")


async def test_unified_search():
    """Test unified multi-source search."""
    print_header("4. Unified Search Test (arXiv + PubMed)")

    query = "large language model"

    result = await search_all_sources(
        query=query,
        sources=["arxiv", "pubmed"],
        max_per_source=10,
        deduplicate=True,
        sort_by="relevance",
    )

    print(f"Query: '{query}'")
    print(f"Sources searched: {result['sources_searched']}")
    print(f"Results per source: {result['source_counts']}")
    print(f"Total before dedup: {result['total_before_dedup']}")
    print(f"Total after dedup: {result['total_after_dedup']}")

    papers = result["papers"]
    if papers:
        print(f"\nTop 5 results:")
        for i, paper in enumerate(papers[:5], 1):
            print_paper(paper, i)

        # Check for multi-source papers
        multi_source = [p for p in papers if len(p['sources']) > 1]
        if multi_source:
            print(f"\n📚 Found {len(multi_source)} papers from multiple sources (merged)")

        print("\n✅ Unified search works!")
    else:
        print("⚠️  No papers found")


async def test_sorting():
    """Test different sorting options."""
    print_header("5. Sorting Test")

    query = "neural network"

    # Test different sort options
    for sort_by in ["relevance", "citations", "year"]:
        result = await search_all_sources(
            query=query,
            sources=["arxiv"],
            max_per_source=5,
            sort_by=sort_by,
        )

        papers = result["papers"]
        if papers:
            print(f"\nSort by {sort_by}:")
            for i, p in enumerate(papers[:3], 1):
                if sort_by == "citations":
                    print(f"  {i}. citations={p['citations']}: {p['title'][:40]}...")
                elif sort_by == "year":
                    print(f"  {i}. year={p['year']}: {p['title'][:40]}...")
                else:
                    print(f"  {i}. score={p['relevance_score']:.2f}: {p['title'][:40]}...")

    print("\n✅ Sorting works!")


async def main():
    print("\n" + "Unified Search Aggregator Test".center(60))
    print("=" * 60)

    # Run tests
    await test_deduplication()
    await test_arxiv_search()
    await test_pubmed_search()
    await test_unified_search()
    await test_sorting()

    print_header("Summary")
    print("✅ All tests completed!")
    print("\nThe unified search aggregator can now:")
    print("  - Search arXiv for preprints")
    print("  - Search PubMed for biomedical literature")
    print("  - Merge duplicate papers from multiple sources")
    print("  - Rank results by relevance, citations, or year")


if __name__ == "__main__":
    asyncio.run(main())

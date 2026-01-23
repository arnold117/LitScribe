#!/usr/bin/env python3
"""Test script for PDF Parser MCP Server functions."""

import os
# Fix OpenMP duplicate library issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_servers.pdf_parser_server import (
    parse_pdf,
    extract_section,
    extract_all_tables,
    extract_all_citations,
    get_document_info,
    clear_cache,
)
from utils.config import Config


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def find_test_pdf() -> str | None:
    """Find a PDF file for testing."""
    # Check common locations
    search_paths = [
        Config.DATA_DIR / "pdfs",
        Path.home() / "Downloads",
        Path.home() / "Documents",
    ]

    for search_path in search_paths:
        if search_path.exists():
            pdfs = list(search_path.glob("*.pdf"))
            if pdfs:
                return str(pdfs[0])

    return None


async def main():
    print("\n" + "PDF Parser MCP Server Test".center(60))
    print("=" * 60)

    # Find a test PDF
    pdf_path = find_test_pdf()

    if not pdf_path:
        print("\n⚠️  No PDF file found for testing.")
        print("\nTo test, either:")
        print("  1. Download a paper using arXiv MCP: ")
        print("     python scripts/test_arxiv.py")
        print("  2. Place a PDF in ./data/pdfs/")
        print("  3. Provide a path directly in this script")

        # Create a simple test with a non-existent file to test error handling
        print_header("Error Handling Test")
        result = await get_document_info("/nonexistent/file.pdf")
        print(f"Error handling works: {'error' in result}")
        print(f"Error message: {result.get('error', 'N/A')}")
        return

    print(f"\nTest PDF: {pdf_path}")
    print(f"File size: {Path(pdf_path).stat().st_size / (1024*1024):.2f} MB")

    # Test 1: Get document info (fast)
    print_header("1. Get Document Info (Fast)")
    info = await get_document_info(pdf_path)
    print(f"File: {info.get('file_name')}")
    print(f"Pages: {info.get('num_pages')}")
    print(f"Title: {info.get('title', '(none)')[:50]}")
    print(f"Author: {info.get('author', '(none)')[:50]}")
    print(f"Cached: {info.get('cached')}")
    if info.get("sections"):
        print(f"Sections: {len(info['sections'])}")
        for s in info["sections"][:5]:
            print(f"  - {s[:40]}...")

    # Test 2: Full PDF parse
    print_header("2. Full PDF Parse (may take 30-60s first time)")
    print("Parsing...")
    result = await parse_pdf(pdf_path)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"✅ Parse complete!")
    print(f"Word count: {result['metadata'].get('word_count', 0)}")
    print(f"Sections: {len(result['sections'])}")
    print(f"Tables: {len(result['tables'])}")
    print(f"Equations: {len(result['equations'])}")
    print(f"Citations: {len(result['citations'])}")
    print(f"References: {len(result['references'])}")

    # Show first part of markdown
    print("\nMarkdown preview (first 500 chars):")
    print("-" * 40)
    print(result["markdown"][:500])
    print("...")

    # Test 3: Extract specific section
    print_header("3. Extract Section")
    section_names = ["abstract", "introduction", "method", "conclusion"]
    for name in section_names:
        section = await extract_section(pdf_path, name)
        if section.get("found"):
            print(f"✅ Found '{section['title']}'")
            print(f"   Content preview: {section['content'][:100]}...")
            break
    else:
        print("Available sections:")
        for s in result["sections"][:5]:
            print(f"  - {s['title']}")

    # Test 4: Extract tables
    print_header("4. Extract Tables")
    tables_result = await extract_all_tables(pdf_path)
    print(f"Found {tables_result['count']} tables")
    for i, table in enumerate(tables_result["tables"][:2], 1):
        print(f"\n  Table {i}: {table['caption'][:50]}")
        # Show first few lines of table content
        lines = table["content"].split("\n")[:3]
        for line in lines:
            print(f"    {line[:60]}...")

    # Test 5: Extract citations
    print_header("5. Extract Citations")
    citations_result = await extract_all_citations(pdf_path)
    print(f"In-text citations: {citations_result['citation_count']}")
    print(f"Bibliography entries: {citations_result['reference_count']}")

    if citations_result["citations"]:
        print("\nSample citations:")
        for c in citations_result["citations"][:3]:
            print(f"  - {c['text']} in context: ...{c['context'][-50:]}...")

    if citations_result["references"]:
        print("\nSample references:")
        for r in citations_result["references"][:3]:
            print(f"  - {r[:70]}...")

    # Test 6: Cache verification
    print_header("6. Cache Verification")
    info_after = await get_document_info(pdf_path)
    print(f"Cached after parse: {info_after.get('cached')}")
    if info_after.get("cached"):
        print(f"Cached sections: {len(info_after.get('sections', []))}")
        print(f"Cached word count: {info_after.get('word_count')}")

    print_header("Summary")
    print("✅ All PDF Parser MCP functions tested successfully!")


if __name__ == "__main__":
    asyncio.run(main())

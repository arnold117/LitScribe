#!/usr/bin/env python3
"""Test script for Zotero MCP Server functions."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_servers.zotero_server import (
    search_items,
    get_item_metadata,
    get_collection_items,
    get_collections,
    get_item_pdf_path,
    get_recent_items,
)
from utils.config import Config


def print_header(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}\n")


async def main():
    print("\n" + "Zotero MCP Server Test".center(50))
    print("=" * 50)

    print(f"\nLibrary ID: {Config.ZOTERO_LIBRARY_ID}")
    print(f"Default Collection: {Config.ZOTERO_DEFAULT_COLLECTION}")
    print(f"Storage Dir: {Config.ZOTERO_STORAGE_DIR}")

    # Test 1: Get collections
    print_header("1. Get Collections")
    collections = await get_collections()
    print(f"Found {collections['count']} collections:")
    for col in collections["collections"][:5]:
        print(f"  - [{col['key']}] {col['name']} ({col['item_count']} items)")

    # Test 2: Get collection items
    print_header("2. Get Collection Items")
    items = await get_collection_items(limit=5)
    print(f"Collection: {items['collection_name']}")
    print(f"Found {items['count']} items:")
    for item in items["items"]:
        authors = ", ".join(item["creators"][0].get("lastName", "") for item in [item] if item["creators"])
        print(f"  - [{item['item_type']}] {item['title'][:50]}...")
        if item["doi"]:
            print(f"    DOI: {item['doi']}")

    # Test 3: Get recent items
    print_header("3. Get Recent Items")
    recent = await get_recent_items(limit=3)
    print(f"Recent {recent['count']} items:")
    for item in recent["items"]:
        print(f"  - {item['title'][:50]}...")
        print(f"    Date: {item['date']}")

    # Test 4: Search items
    print_header("4. Search Items")
    if items["items"]:
        # Search for first word of first item's title
        first_title = items["items"][0]["title"]
        search_term = first_title.split()[0] if first_title else "test"
        search_results = await search_items(search_term, limit=3)
        print(f"Search '{search_term}': {search_results['count']} results")
        for item in search_results["items"][:3]:
            print(f"  - {item['title'][:50]}...")

    # Test 5: Get item metadata
    print_header("5. Get Item Metadata")
    if items["items"]:
        item_key = items["items"][0]["key"]
        metadata = await get_item_metadata(item_key)
        print(f"Item Key: {metadata['key']}")
        print(f"Title: {metadata['title']}")
        print(f"Type: {metadata['item_type']}")
        print(f"Abstract: {metadata['abstract'][:100]}..." if metadata["abstract"] else "Abstract: (none)")
        if metadata.get("pdf_attachments"):
            print(f"PDF Attachments: {len(metadata['pdf_attachments'])}")
            for pdf in metadata["pdf_attachments"]:
                print(f"  - {pdf['filename']}")
                print(f"    Local: {pdf['local_path']}")

    # Test 6: Get PDF path
    print_header("6. Get PDF Path")
    if items["items"]:
        item_key = items["items"][0]["key"]
        pdf_info = await get_item_pdf_path(item_key)
        if pdf_info.get("local_path"):
            print(f"Item: {item_key}")
            print(f"Filename: {pdf_info['filename']}")
            print(f"Local Path: {pdf_info['local_path']}")
            print(f"Exists: {pdf_info['exists']}")
        else:
            print(f"No PDF found for item {item_key}")
            if "error" in pdf_info:
                print(f"Error: {pdf_info['error']}")

    print_header("Summary")
    print("All Zotero MCP functions tested successfully!")


if __name__ == "__main__":
    asyncio.run(main())

"""Zotero MCP Server - Access and manage Zotero library via Cloud API."""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyzotero import zotero

from models.zotero_item import ZoteroItem, Fulltext
from utils.config import Config


def _get_zotero_client() -> zotero.Zotero:
    """Create Zotero API client."""
    return zotero.Zotero(
        library_id=Config.ZOTERO_LIBRARY_ID,
        library_type=Config.ZOTERO_LIBRARY_TYPE,
        api_key=Config.ZOTERO_API_KEY,
    )


def _parse_zotero_item(item: dict) -> ZoteroItem:
    """Parse Zotero API response into ZoteroItem."""
    data = item.get("data", {})

    # Extract tags
    tags = [t.get("tag", "") for t in data.get("tags", [])]

    # Extract collections
    collections = data.get("collections", [])

    # Build ZoteroItem
    return ZoteroItem(
        key=data.get("key", ""),
        item_type=data.get("itemType", ""),
        title=data.get("title", ""),
        creators=data.get("creators", []),
        abstract=data.get("abstractNote", ""),
        date=data.get("date", ""),
        url=data.get("url", ""),
        tags=tags,
        collections=collections,
        publication_title=data.get("publicationTitle", ""),
        volume=data.get("volume", ""),
        issue=data.get("issue", ""),
        pages=data.get("pages", ""),
        doi=data.get("DOI"),
        extra=data.get("extra", {}),
    )


def _get_pdf_path(attachment_key: str, filename: str) -> Optional[str]:
    """Construct local PDF path from attachment key."""
    storage_dir = Config.ZOTERO_STORAGE_DIR
    pdf_path = storage_dir / attachment_key / filename
    if pdf_path.exists():
        return str(pdf_path)
    return None



async def search_items(
    query: str,
    collection: Optional[str] = None,
    item_type: Optional[str] = None,
    limit: int = 25,
) -> dict:
    """
    Search Zotero library for items.

    Args:
        query: Search query (searches title, creators, tags)
        collection: Collection key to search within (default: ZOTERO_DEFAULT_COLLECTION)
        item_type: Filter by item type (e.g., "journalArticle", "book")
        limit: Maximum number of results (default: 25)

    Returns:
        Dictionary with count and list of matching items
    """
    loop = asyncio.get_event_loop()

    # Use default collection if not specified
    if collection is None:
        collection = Config.ZOTERO_DEFAULT_COLLECTION

    def do_search():
        zot = _get_zotero_client()

        # Build search parameters
        params = {"q": query, "limit": limit}
        if item_type:
            params["itemType"] = item_type

        # Search within collection or entire library
        if collection:
            items = zot.collection_items(collection, **params)
        else:
            items = zot.items(**params)

        return items

    items = await loop.run_in_executor(None, do_search)

    # Filter out attachments and notes
    items = [i for i in items if i.get("data", {}).get("itemType") not in ("attachment", "note")]

    parsed = [_parse_zotero_item(i) for i in items]

    return {
        "count": len(parsed),
        "collection": collection,
        "items": [p.to_dict() for p in parsed],
    }



async def get_item_metadata(item_key: str) -> dict:
    """
    Get detailed metadata for a specific Zotero item.

    Args:
        item_key: Zotero item key

    Returns:
        Full item metadata including attachments info
    """
    loop = asyncio.get_event_loop()

    def do_fetch():
        zot = _get_zotero_client()
        item = zot.item(item_key)
        return item

    item = await loop.run_in_executor(None, do_fetch)

    if not item:
        return {"error": f"Item {item_key} not found"}

    parsed = _parse_zotero_item(item)

    # Also get attachments
    def get_attachments():
        zot = _get_zotero_client()
        children = zot.children(item_key)
        return children

    children = await loop.run_in_executor(None, get_attachments)

    # Find PDF attachments
    pdf_attachments = []
    for child in children:
        child_data = child.get("data", {})
        if child_data.get("itemType") == "attachment":
            content_type = child_data.get("contentType", "")
            if "pdf" in content_type.lower():
                attachment_key = child_data.get("key", "")
                filename = child_data.get("filename", "")
                local_path = _get_pdf_path(attachment_key, filename)
                pdf_attachments.append({
                    "key": attachment_key,
                    "filename": filename,
                    "local_path": local_path,
                })

    result = parsed.to_dict()
    result["pdf_attachments"] = pdf_attachments

    return result



async def get_collection_items(
    collection_key: Optional[str] = None,
    limit: int = 50,
    include_subcollections: bool = False,
) -> dict:
    """
    Get all items in a Zotero collection.

    Args:
        collection_key: Collection key (default: ZOTERO_DEFAULT_COLLECTION)
        limit: Maximum number of items (default: 50)
        include_subcollections: Include items from subcollections

    Returns:
        Dictionary with collection info and items
    """
    loop = asyncio.get_event_loop()

    if collection_key is None:
        collection_key = Config.ZOTERO_DEFAULT_COLLECTION

    def do_fetch():
        zot = _get_zotero_client()

        # Get collection info
        collection_info = zot.collection(collection_key) if collection_key else None

        # Get items
        if collection_key:
            items = zot.collection_items(collection_key, limit=limit)
        else:
            items = zot.top(limit=limit)

        return collection_info, items

    collection_info, items = await loop.run_in_executor(None, do_fetch)

    # Filter out attachments and notes
    items = [i for i in items if i.get("data", {}).get("itemType") not in ("attachment", "note")]

    parsed = [_parse_zotero_item(i) for i in items]

    collection_name = ""
    if collection_info:
        collection_name = collection_info.get("data", {}).get("name", "")

    return {
        "collection_key": collection_key,
        "collection_name": collection_name,
        "count": len(parsed),
        "items": [p.to_dict() for p in parsed],
    }



async def get_collections(parent_collection: Optional[str] = None) -> dict:
    """
    Get list of collections in the library.

    Args:
        parent_collection: Parent collection key to list subcollections

    Returns:
        List of collections with their keys and names
    """
    loop = asyncio.get_event_loop()

    def do_fetch():
        zot = _get_zotero_client()
        if parent_collection:
            collections = zot.collections_sub(parent_collection)
        else:
            collections = zot.collections()
        return collections

    collections = await loop.run_in_executor(None, do_fetch)

    result = []
    for col in collections:
        data = col.get("data", {})
        result.append({
            "key": data.get("key", ""),
            "name": data.get("name", ""),
            "parent": data.get("parentCollection", None),
            "item_count": col.get("meta", {}).get("numItems", 0),
        })

    return {
        "count": len(result),
        "collections": result,
    }



async def get_item_pdf_path(item_key: str) -> dict:
    """
    Get local PDF file path for a Zotero item.

    Args:
        item_key: Zotero item key

    Returns:
        Dictionary with PDF path info
    """
    loop = asyncio.get_event_loop()

    def do_fetch():
        zot = _get_zotero_client()
        children = zot.children(item_key)
        return children

    children = await loop.run_in_executor(None, do_fetch)

    for child in children:
        child_data = child.get("data", {})
        if child_data.get("itemType") == "attachment":
            content_type = child_data.get("contentType", "")
            if "pdf" in content_type.lower():
                attachment_key = child_data.get("key", "")
                filename = child_data.get("filename", "")
                local_path = _get_pdf_path(attachment_key, filename)

                return {
                    "item_key": item_key,
                    "attachment_key": attachment_key,
                    "filename": filename,
                    "local_path": local_path,
                    "exists": local_path is not None,
                }

    return {
        "item_key": item_key,
        "error": "No PDF attachment found",
    }



async def add_note(
    parent_key: str,
    note_content: str,
    tags: Optional[List[str]] = None,
) -> dict:
    """
    Add a note to a Zotero item.

    Args:
        parent_key: Parent item key
        note_content: HTML content of the note
        tags: Optional list of tags

    Returns:
        Created note info
    """
    loop = asyncio.get_event_loop()

    def do_create():
        zot = _get_zotero_client()

        note_template = zot.item_template("note")
        note_template["note"] = note_content
        note_template["parentItem"] = parent_key

        if tags:
            note_template["tags"] = [{"tag": t} for t in tags]

        result = zot.create_items([note_template])
        return result

    result = await loop.run_in_executor(None, do_create)

    if result.get("successful"):
        created = result["successful"]["0"]
        return {
            "success": True,
            "note_key": created.get("key", ""),
            "parent_key": parent_key,
        }
    else:
        return {
            "success": False,
            "error": result.get("failed", "Unknown error"),
        }



async def get_recent_items(limit: int = 20, collection: Optional[str] = None) -> dict:
    """
    Get recently added/modified items.

    Args:
        limit: Maximum number of items (default: 20)
        collection: Collection key to filter (default: ZOTERO_DEFAULT_COLLECTION)

    Returns:
        List of recent items
    """
    loop = asyncio.get_event_loop()

    if collection is None:
        collection = Config.ZOTERO_DEFAULT_COLLECTION

    def do_fetch():
        zot = _get_zotero_client()

        if collection:
            items = zot.collection_items(collection, limit=limit, sort="dateModified", direction="desc")
        else:
            items = zot.top(limit=limit, sort="dateModified", direction="desc")

        return items

    items = await loop.run_in_executor(None, do_fetch)

    # Filter out attachments and notes
    items = [i for i in items if i.get("data", {}).get("itemType") not in ("attachment", "note")]

    parsed = [_parse_zotero_item(i) for i in items]

    return {
        "count": len(parsed),
        "items": [p.to_dict() for p in parsed],
    }



async def add_item_by_identifier(
    identifier: str,
    collection: Optional[str] = None,
) -> dict:
    """
    Add an item to Zotero by DOI, ISBN, PMID, or arXiv ID.

    Args:
        identifier: DOI, ISBN, PMID, or arXiv ID
        collection: Collection to add to (default: ZOTERO_DEFAULT_COLLECTION)

    Returns:
        Created item info
    """
    loop = asyncio.get_event_loop()

    if collection is None:
        collection = Config.ZOTERO_DEFAULT_COLLECTION

    def do_add():
        zot = _get_zotero_client()

        # Zotero can translate identifiers
        # This requires the translation server, so we'll use a simpler approach
        # by creating a basic item and letting Zotero fetch metadata

        # For now, create via DOI lookup using pyzotero's magic
        try:
            # Try to add by identifier (DOI works best)
            result = zot.add_by_id(identifier)
            return result
        except Exception as e:
            return {"error": str(e)}

    result = await loop.run_in_executor(None, do_add)

    if "error" in result:
        return result

    # Add to collection if specified
    if collection and result.get("successful"):
        def add_to_collection():
            zot = _get_zotero_client()
            item_key = result["successful"]["0"]["key"]
            zot.addto_collection(collection, [{"key": item_key}])
            return item_key

        try:
            item_key = await loop.run_in_executor(None, add_to_collection)
            return {
                "success": True,
                "item_key": item_key,
                "collection": collection,
                "identifier": identifier,
            }
        except Exception as e:
            return {
                "success": True,
                "item_key": result["successful"]["0"]["key"],
                "warning": f"Added but failed to add to collection: {e}",
            }

    return {
        "success": bool(result.get("successful")),
        "result": result,
    }



async def create_or_get_collection(
    name: str,
    parent_collection: Optional[str] = None,
) -> dict:
    """Create a Zotero collection or return it if it already exists.

    Args:
        name: Collection name (e.g., "LitScribe")
        parent_collection: Parent collection key (optional)

    Returns:
        Dictionary with collection key and name
    """
    loop = asyncio.get_event_loop()

    def do_create():
        zot = _get_zotero_client()

        # Check if collection already exists
        if parent_collection:
            existing = zot.collections_sub(parent_collection)
        else:
            existing = zot.collections()

        for col in existing:
            if col.get("data", {}).get("name") == name:
                return {
                    "key": col["data"]["key"],
                    "name": name,
                    "created": False,
                }

        # Create new collection
        payload = {"name": name}
        if parent_collection:
            payload["parentCollection"] = parent_collection

        result = zot.create_collections([payload])
        if result.get("successful"):
            created = result["successful"]["0"]
            return {
                "key": created.get("key", ""),
                "name": name,
                "created": True,
            }
        return {"error": result.get("failed", "Unknown error")}

    return await loop.run_in_executor(None, do_create)



async def save_papers_to_collection(
    papers: List[dict],
    collection_key: Optional[str] = None,
) -> dict:
    """Save papers to a Zotero collection by their identifiers (DOI/arXiv ID).

    Args:
        papers: List of paper dicts with at least 'doi' or 'arxiv_id' field
        collection_key: Target collection key (default: ZOTERO_DEFAULT_COLLECTION)

    Returns:
        Dictionary with success/failure counts
    """
    loop = asyncio.get_event_loop()

    if collection_key is None:
        collection_key = Config.ZOTERO_DEFAULT_COLLECTION

    def do_save():
        zot = _get_zotero_client()

        saved = 0
        failed = 0
        skipped = 0
        results = []

        for paper in papers:
            identifier = paper.get("doi") or paper.get("arxiv_id")
            if not identifier:
                skipped += 1
                continue

            try:
                result = zot.add_by_id(identifier)
                if result.get("successful"):
                    item_key = result["successful"]["0"]["key"]
                    if collection_key:
                        try:
                            zot.addto_collection(collection_key, [{"key": item_key}])
                        except Exception:
                            pass
                    saved += 1
                    results.append({
                        "identifier": identifier,
                        "item_key": item_key,
                        "status": "saved",
                    })
                else:
                    failed += 1
                    results.append({
                        "identifier": identifier,
                        "status": "failed",
                        "error": str(result.get("failed", "")),
                    })
            except Exception as e:
                failed += 1
                results.append({
                    "identifier": identifier,
                    "status": "failed",
                    "error": str(e),
                })

        return {
            "saved": saved,
            "failed": failed,
            "skipped": skipped,
            "total": len(papers),
            "collection_key": collection_key,
            "details": results,
        }

    return await loop.run_in_executor(None, do_save)



async def import_collection_papers(
    collection_key: Optional[str] = None,
    limit: int = 100,
) -> dict:
    """Import all papers from a Zotero collection as unified paper dicts.

    Used to seed a literature review from an existing Zotero collection.

    Args:
        collection_key: Zotero collection key (default: ZOTERO_DEFAULT_COLLECTION)
        limit: Maximum papers to import

    Returns:
        Dictionary with papers in unified format
    """
    import re as _re

    loop = asyncio.get_event_loop()

    if collection_key is None:
        collection_key = Config.ZOTERO_DEFAULT_COLLECTION

    def do_import():
        zot = _get_zotero_client()

        if collection_key:
            items = zot.collection_items(collection_key, limit=limit)
        else:
            items = zot.top(limit=limit)

        return items

    items = await loop.run_in_executor(None, do_import)

    # Filter out attachments and notes
    items = [i for i in items if i.get("data", {}).get("itemType") not in ("attachment", "note")]

    parsed = [_parse_zotero_item(i) for i in items]

    # Convert to unified paper format
    papers = []
    for item in parsed:
        item_dict = item.to_dict()

        authors = []
        for creator in item_dict.get("creators", []):
            if creator.get("creatorType") == "author":
                name = f"{creator.get('firstName', '')} {creator.get('lastName', '')}".strip()
                if name:
                    authors.append(name)

        year = 0
        date_str = item_dict.get("date", "")
        if date_str:
            year_match = _re.search(r"(\d{4})", date_str)
            if year_match:
                year = int(year_match.group(1))

        paper_id = item_dict.get("doi") or f"zotero:{item_dict.get('key', '')}"

        papers.append({
            "paper_id": paper_id,
            "title": item_dict.get("title", ""),
            "authors": authors,
            "abstract": item_dict.get("abstract", ""),
            "year": year,
            "venue": item_dict.get("publication_title", ""),
            "citations": 0,
            "doi": item_dict.get("doi"),
            "arxiv_id": item_dict.get("arxiv_id"),
            "url": item_dict.get("url", ""),
            "source": "zotero",
            "zotero_key": item_dict.get("key", ""),
            "search_origin": "zotero_import",
        })

    collection_name = ""
    if collection_key:
        try:
            def get_name():
                zot = _get_zotero_client()
                col = zot.collection(collection_key)
                return col.get("data", {}).get("name", "")
            collection_name = await loop.run_in_executor(None, get_name)
        except Exception:
            pass

    return {
        "collection_key": collection_key,
        "collection_name": collection_name,
        "count": len(papers),
        "papers": papers,
    }



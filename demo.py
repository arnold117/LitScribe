#!/usr/bin/env python3
"""Demo script to test API connections for LitScribe."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import Config


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}\n")


def print_status(name: str, success: bool, message: str = "") -> None:
    """Print status with emoji."""
    status = "✅" if success else "❌"
    print(f"{status} {name}: {message}")


async def test_deepseek() -> bool:
    """Test DeepSeek API connection."""
    print_header("Testing DeepSeek API")

    try:
        import httpx

        # Direct HTTP call to DeepSeek API (more reliable for testing)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "Say 'Hello LitScribe!' in one line."}],
                    "max_tokens": 50,
                },
            )

            if response.status_code == 200:
                data = response.json()
                reply = data["choices"][0]["message"]["content"].strip()
                print_status("DeepSeek", True, f"Response: {reply}")
                return True
            else:
                print_status("DeepSeek", False, f"HTTP {response.status_code}: {response.text[:100]}")
                return False

    except Exception as e:
        print_status("DeepSeek", False, str(e))
        return False


async def test_pubmed() -> bool:
    """Test PubMed API connection."""
    print_header("Testing PubMed API")

    try:
        from Bio import Entrez

        # Set required email
        Entrez.email = Config.NCBI_EMAIL
        if Config.NCBI_API_KEY:
            Entrez.api_key = Config.NCBI_API_KEY

        # Search for a simple query
        handle = Entrez.esearch(
            db="pubmed",
            term="machine learning",
            retmax=3,
        )
        record = Entrez.read(handle)
        handle.close()

        count = record.get("Count", 0)
        ids = record.get("IdList", [])

        print_status("PubMed Search", True, f"Found {count} results, first 3 IDs: {ids}")

        # Fetch one article's metadata
        if ids:
            handle = Entrez.efetch(db="pubmed", id=ids[0], rettype="xml")
            from Bio import Medline
            # Use efetch with rettype=medline for simpler parsing
            handle.close()

            handle = Entrez.efetch(db="pubmed", id=ids[0], rettype="medline", retmode="text")
            records = list(Medline.parse(handle))
            handle.close()

            if records:
                article = records[0]
                title = article.get("TI", "No title")
                print_status("PubMed Fetch", True, f"Title: {title[:60]}...")

        return True

    except Exception as e:
        print_status("PubMed", False, str(e))
        return False


async def test_zotero() -> bool:
    """Test Zotero API connection."""
    print_header("Testing Zotero API")

    try:
        from pyzotero import zotero

        zot = zotero.Zotero(
            library_id=Config.ZOTERO_LIBRARY_ID,
            library_type=Config.ZOTERO_LIBRARY_TYPE,
            api_key=Config.ZOTERO_API_KEY,
        )

        # Get library info
        items = zot.top(limit=5)

        print_status("Zotero Connection", True, f"Found {len(items)} items in library")

        # Show first few items
        if items:
            print("\n  Recent items:")
            for item in items[:3]:
                title = item.get("data", {}).get("title", "Untitled")
                item_type = item.get("data", {}).get("itemType", "unknown")
                print(f"    - [{item_type}] {title[:50]}...")

        return True

    except Exception as e:
        print_status("Zotero", False, str(e))
        return False


async def main():
    """Run all API tests."""
    print("\n" + "🔬 LitScribe API Connection Test".center(50))
    print("=" * 50)

    # Show current configuration
    print("\n📋 Configuration:")
    keys = Config.validate_api_keys()
    for name, configured in keys.items():
        status = "✅" if configured else "⚠️  (not set)"
        print(f"   {name}: {status}")

    print(f"\n   LiteLLM Model: {Config.LITELLM_MODEL}")
    print(f"   NCBI Email: {Config.NCBI_EMAIL}")
    print(f"   Zotero Library: {Config.ZOTERO_LIBRARY_ID} ({Config.ZOTERO_LIBRARY_TYPE})")

    # Run tests
    results = {}

    if Config.DEEPSEEK_API_KEY:
        results["DeepSeek"] = await test_deepseek()
    else:
        print_header("Testing DeepSeek API")
        print_status("DeepSeek", False, "API key not configured")
        results["DeepSeek"] = False

    if Config.NCBI_EMAIL:
        results["PubMed"] = await test_pubmed()
    else:
        print_header("Testing PubMed API")
        print_status("PubMed", False, "NCBI email not configured")
        results["PubMed"] = False

    if Config.ZOTERO_API_KEY and Config.ZOTERO_LIBRARY_ID:
        results["Zotero"] = await test_zotero()
    else:
        print_header("Testing Zotero API")
        print_status("Zotero", False, "API key or library ID not configured")
        results["Zotero"] = False

    # Summary
    print_header("Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All API connections working!")
    else:
        print("\n⚠️  Some APIs need attention. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

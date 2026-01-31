#!/usr/bin/env python
"""Manual test script for the cache system.

Run with: python tests/test_cache_manual.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_database_init():
    """Test database initialization."""
    print("=" * 60)
    print("Test 1: Database Initialization")
    print("=" * 60)

    from cache.database import CacheDatabase

    # Use temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = CacheDatabase(db_path)
        db.init_schema()

        print(f"Database created at: {db_path}")
        print(f"Database exists: {db_path.exists()}")

        # Check tables
        with db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Tables created: {tables}")

        # Get stats
        stats = db.get_stats()
        print(f"Stats: {stats}")

        expected_tables = ['schema_version', 'papers', 'pdfs', 'parsed_docs',
                          'search_cache', 'llm_cache', 'command_logs']
        for table in expected_tables:
            if table not in tables:
                print(f"FAIL: Missing table {table}")
                return False

    print("PASS: Database initialization")
    return True


def test_paper_cache():
    """Test paper cache operations."""
    print("\n" + "=" * 60)
    print("Test 2: Paper Cache")
    print("=" * 60)

    from cache.database import CacheDatabase
    from cache.paper_cache import PaperCache

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = CacheDatabase(db_path)
        cache = PaperCache(db)

        # Test save and get
        paper = {
            "paper_id": "arxiv:2401.12345",
            "title": "Test Paper on LLMs",
            "authors": ["John Doe", "Jane Smith"],
            "abstract": "This is a test abstract about large language models.",
            "year": 2024,
            "source": "arxiv",
            "citations": 42,
        }

        paper_id = cache.save(paper)
        print(f"Saved paper with ID: {paper_id}")

        # Retrieve
        retrieved = cache.get("arxiv:2401.12345")
        print(f"Retrieved paper: {retrieved['title']}")

        # Check exists
        exists = cache.exists("arxiv:2401.12345")
        print(f"Paper exists: {exists}")

        not_exists = cache.exists("arxiv:9999.99999")
        print(f"Non-existent paper exists: {not_exists}")

        if retrieved["title"] != paper["title"]:
            print("FAIL: Retrieved title doesn't match")
            return False

        if not exists or not_exists:
            print("FAIL: Exists check failed")
            return False

    print("PASS: Paper cache")
    return True


def test_search_cache():
    """Test search cache with TTL."""
    print("\n" + "=" * 60)
    print("Test 3: Search Cache (TTL)")
    print("=" * 60)

    from cache.database import CacheDatabase
    from cache.search_cache import SearchCache

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = CacheDatabase(db_path)
        cache = SearchCache(db, ttl_hours=24)

        # Test save and get
        query = "large language model reasoning"
        source = "arxiv"
        results = [
            {"paper_id": "arxiv:2401.00001", "title": "Paper 1"},
            {"paper_id": "arxiv:2401.00002", "title": "Paper 2"},
        ]

        cache.save(query, source, results, total_found=100)
        print(f"Saved search results for '{query}'")

        # Retrieve
        cached = cache.get(query, source)
        if cached:
            cached_results, total = cached
            print(f"Cache hit: {len(cached_results)} results, total={total}")
        else:
            print("FAIL: Cache miss when expecting hit")
            return False

        # Test cache miss
        miss = cache.get("nonexistent query", "arxiv")
        print(f"Cache miss for new query: {miss is None}")

        if miss is not None:
            print("FAIL: Expected cache miss")
            return False

    print("PASS: Search cache")
    return True


async def test_async_operations():
    """Test async cache operations."""
    print("\n" + "=" * 60)
    print("Test 4: Async Operations")
    print("=" * 60)

    from cache.database import CacheDatabase
    from cache.paper_cache import PaperCache
    from cache.search_cache import SearchCache

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = CacheDatabase(db_path)
        paper_cache = PaperCache(db)
        search_cache = SearchCache(db)

        # Async paper save/get
        paper = {
            "paper_id": "s2:abc123",
            "title": "Async Test Paper",
            "authors": ["Async Author"],
            "abstract": "Testing async operations.",
            "year": 2024,
            "source": "semantic_scholar",
        }

        await paper_cache.save_async(paper)
        retrieved = await paper_cache.get_async("s2:abc123")
        print(f"Async paper save/get: {retrieved['title']}")

        # Async search save/get
        await search_cache.save_async(
            "async test query",
            "semantic_scholar",
            [{"paper_id": "s2:xyz789", "title": "Async Result"}],
        )
        cached = await search_cache.get_async("async test query", "semantic_scholar")
        if cached:
            results, total = cached
            print(f"Async search cache: {len(results)} results")
        else:
            print("FAIL: Async search cache miss")
            return False

    print("PASS: Async operations")
    return True


def test_cache_stats():
    """Test cache statistics."""
    print("\n" + "=" * 60)
    print("Test 5: Cache Statistics")
    print("=" * 60)

    from cache.database import CacheDatabase
    from cache.paper_cache import PaperCache
    from cache.search_cache import SearchCache

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = CacheDatabase(db_path)
        paper_cache = PaperCache(db)
        search_cache = SearchCache(db)

        # Add some data
        for i in range(5):
            paper_cache.save({
                "paper_id": f"arxiv:2401.{i:05d}",
                "title": f"Test Paper {i}",
                "authors": ["Author"],
                "abstract": "Abstract",
                "year": 2024,
                "source": "arxiv",
            })

        search_cache.save("test query", "arxiv", [{"id": "1"}], 10)

        # Get stats
        stats = db.get_stats()
        print(f"Papers count: {stats.get('papers_count')}")
        print(f"Search cache count: {stats.get('search_cache_count')}")
        print(f"DB size: {stats.get('db_size_mb')} MB")

        if stats.get('papers_count') != 5:
            print("FAIL: Expected 5 papers")
            return False

    print("PASS: Cache statistics")
    return True


def main():
    """Run all tests."""
    print("LitScribe Cache System Tests")
    print("=" * 60)

    results = []

    # Sync tests
    results.append(("Database Init", test_database_init()))
    results.append(("Paper Cache", test_paper_cache()))
    results.append(("Search Cache", test_search_cache()))
    results.append(("Cache Stats", test_cache_stats()))

    # Async tests
    results.append(("Async Operations", asyncio.run(test_async_operations())))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Test script for LangGraph checkpointing integration.

Run with: python tests/test_checkpointing.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_imports():
    """Test that checkpointing imports work."""
    print("=" * 60)
    print("Test 1: Checkpointing imports")
    print("=" * 60)

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        print("AsyncSqliteSaver imported successfully")

        from agents.graph import (
            get_checkpoint_db_path,
            run_literature_review,
            resume_literature_review,
            get_review_state,
        )
        print("All graph functions imported successfully")

        print("PASS: Checkpointing imports")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_checkpointer_context_manager():
    """Test that checkpointer context manager works."""
    print("\n" + "=" * 60)
    print("Test 2: Checkpointer context manager")
    print("=" * 60)

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from agents.graph import get_checkpoint_db_path

        # Check the database path
        db_path = get_checkpoint_db_path()
        print(f"Checkpoint database path: {db_path}")

        # Create the checkpointer using context manager
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            print(f"Checkpointer created: {type(checkpointer).__name__}")
            if checkpointer is None:
                print("FAIL: Checkpointer is None")
                return False

        print("PASS: Checkpointer context manager")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_compile_graph_with_checkpointer():
    """Test that graph can be compiled with checkpointer."""
    print("\n" + "=" * 60)
    print("Test 3: Compile graph with checkpointer")
    print("=" * 60)

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from agents.graph import compile_graph, get_checkpoint_db_path

        db_path = get_checkpoint_db_path()

        # Use context manager to create checkpointer
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            # Compile graph with checkpointer
            graph = compile_graph(checkpointer=checkpointer)
            print(f"Graph compiled with checkpointer: {type(graph).__name__}")

            # Check that the graph has nodes
            if hasattr(graph, 'nodes'):
                print(f"Graph has nodes attribute")

        print("PASS: Compile graph with checkpointer")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_state_with_cache():
    """Test that state has cache_enabled field."""
    print("\n" + "=" * 60)
    print("Test 4: State cache_enabled field")
    print("=" * 60)

    try:
        from agents.state import create_initial_state

        # Create state with cache enabled
        state = create_initial_state(
            research_question="test question",
            cache_enabled=True,
        )

        if "cache_enabled" not in state:
            print("FAIL: cache_enabled not in state")
            return False

        if state["cache_enabled"] != True:
            print("FAIL: cache_enabled should be True")
            return False

        print(f"State has cache_enabled: {state['cache_enabled']}")

        print("PASS: State cache_enabled field")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_run_function_signature():
    """Test that run_literature_review has checkpoint parameters."""
    print("\n" + "=" * 60)
    print("Test 5: run_literature_review signature")
    print("=" * 60)

    try:
        from agents.graph import run_literature_review
        import inspect

        sig = inspect.signature(run_literature_review)
        params = list(sig.parameters.keys())
        print(f"Parameters: {params}")

        required_params = [
            "thread_id", "checkpoint_enabled", "cache_enabled",
            # Phase 9.5 additions
            "disable_self_review", "disable_domain_filter", "disable_snowball",
            # Zotero collection passthrough
            "zotero_collection",
        ]
        for param in required_params:
            if param not in params:
                print(f"FAIL: {param} parameter not found")
                return False
            print(f"  - {param}: OK")

        print("PASS: run_literature_review signature")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ablation_flags_in_state():
    """Test that Phase 9.5 ablation flags are in state. (Phase 9.5)"""
    print("\n" + "=" * 60)
    print("Test 6: Ablation flags in state")
    print("=" * 60)

    try:
        from agents.state import create_initial_state

        # Default: all ablation flags False
        state = create_initial_state(research_question="test question")
        for flag in ["disable_self_review", "disable_domain_filter", "disable_snowball"]:
            if flag not in state:
                print(f"FAIL: {flag} not in state")
                return False
            if state[flag] is not False:
                print(f"FAIL: {flag} should default to False, got {state[flag]}")
                return False
            print(f"  - {flag}: {state[flag]} (OK)")

        # Explicit: set ablation flags
        state2 = create_initial_state(
            research_question="test",
            disable_self_review=True,
            disable_domain_filter=True,
            disable_snowball=True,
        )
        for flag in ["disable_self_review", "disable_domain_filter", "disable_snowball"]:
            if state2[flag] is not True:
                print(f"FAIL: {flag} should be True, got {state2[flag]}")
                return False

        # Token tracker moved to ContextVar (no longer in state)
        if "token_tracker" in state:
            print("FAIL: token_tracker should NOT be in state (moved to ContextVar)")
            return False
        print(f"  - token_tracker: not in state (moved to ContextVar, OK)")

        print("PASS: Ablation flags in state")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("LangGraph Checkpointing Integration Tests")
    print("=" * 60)

    results = []

    results.append(("Checkpointing imports", await test_imports()))
    results.append(("Checkpointer context manager", await test_checkpointer_context_manager()))
    results.append(("Compile with checkpointer", await test_compile_graph_with_checkpointer()))
    results.append(("State cache_enabled", await test_state_with_cache()))
    results.append(("run_literature_review signature", await test_run_function_signature()))
    results.append(("Ablation flags in state", await test_ablation_flags_in_state()))

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
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""Tests for TokenTracker (Phase 9.5 Step 1).

Tests cover:
- Basic initialization
- Recording token usage
- Per-agent and per-model breakdown
- Cost estimation with different models
- CLI summary formatting
- Edge cases (empty tracker, unknown model)
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_init():
    """TokenTracker initializes with empty records."""
    from utils.token_tracker import TokenTracker

    tracker = TokenTracker()
    summary = tracker.summary()

    assert summary["total_tokens"] == 0
    assert summary["total_prompt_tokens"] == 0
    assert summary["total_completion_tokens"] == 0
    assert summary["total_calls"] == 0
    assert summary["estimated_cost_usd"] == 0.0
    assert summary["by_agent"] == {}
    assert summary["by_model"] == {}
    print("PASS: test_init")


def test_single_record():
    """Record a single LLM call and verify summary."""
    from utils.token_tracker import TokenTracker

    tracker = TokenTracker()
    tracker.record("discovery", "claude-sonnet-4-5", {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
    })

    s = tracker.summary()
    assert s["total_tokens"] == 1500
    assert s["total_prompt_tokens"] == 1000
    assert s["total_completion_tokens"] == 500
    assert s["total_calls"] == 1

    assert "discovery" in s["by_agent"]
    assert s["by_agent"]["discovery"]["prompt_tokens"] == 1000
    assert s["by_agent"]["discovery"]["completion_tokens"] == 500
    assert s["by_agent"]["discovery"]["calls"] == 1

    assert "claude-sonnet-4-5" in s["by_model"]
    assert s["by_model"]["claude-sonnet-4-5"]["calls"] == 1
    print("PASS: test_single_record")


def test_multiple_agents():
    """Record calls from multiple agents and verify breakdown."""
    from utils.token_tracker import TokenTracker

    tracker = TokenTracker()
    tracker.record("discovery", "claude-sonnet-4-5", {
        "prompt_tokens": 100, "completion_tokens": 50,
    })
    tracker.record("synthesis", "claude-sonnet-4-5", {
        "prompt_tokens": 200, "completion_tokens": 100,
    })
    tracker.record("self_review", "claude-sonnet-4-5", {
        "prompt_tokens": 150, "completion_tokens": 75,
    })

    s = tracker.summary()
    assert s["total_tokens"] == 675  # 100+50 + 200+100 + 150+75
    assert s["total_calls"] == 3
    assert len(s["by_agent"]) == 3
    assert s["by_agent"]["discovery"]["calls"] == 1
    assert s["by_agent"]["synthesis"]["calls"] == 1
    assert s["by_agent"]["self_review"]["calls"] == 1
    print("PASS: test_multiple_agents")


def test_multiple_models():
    """Record calls with different models and verify per-model breakdown."""
    from utils.token_tracker import TokenTracker

    tracker = TokenTracker()
    tracker.record("synthesis", "claude-opus-4-5", {
        "prompt_tokens": 500, "completion_tokens": 1000,
    })
    tracker.record("discovery", "claude-sonnet-4-5", {
        "prompt_tokens": 300, "completion_tokens": 200,
    })

    s = tracker.summary()
    assert len(s["by_model"]) == 2
    assert "claude-opus-4-5" in s["by_model"]
    assert "claude-sonnet-4-5" in s["by_model"]
    assert s["by_model"]["claude-opus-4-5"]["calls"] == 1
    assert s["by_model"]["claude-sonnet-4-5"]["calls"] == 1
    print("PASS: test_multiple_models")


def test_cost_estimation():
    """Verify cost calculation with known pricing."""
    from utils.token_tracker import TokenTracker, MODEL_PRICING

    tracker = TokenTracker()
    # 1M prompt tokens + 1M completion tokens with sonnet
    tracker.record("test", "claude-sonnet-4-5", {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
    })

    s = tracker.summary()
    pricing = MODEL_PRICING["claude-sonnet-4-5"]
    expected_cost = pricing["input"] + pricing["output"]  # 3.0 + 15.0
    assert abs(s["estimated_cost_usd"] - expected_cost) < 0.01, \
        f"Expected ~{expected_cost}, got {s['estimated_cost_usd']}"
    print("PASS: test_cost_estimation")


def test_cost_opus_vs_sonnet():
    """Opus should cost more than Sonnet for same usage."""
    from utils.token_tracker import TokenTracker

    tracker1 = TokenTracker()
    tracker1.record("test", "claude-opus-4-5", {
        "prompt_tokens": 1000, "completion_tokens": 1000,
    })

    tracker2 = TokenTracker()
    tracker2.record("test", "claude-sonnet-4-5", {
        "prompt_tokens": 1000, "completion_tokens": 1000,
    })

    opus_cost = tracker1.summary()["estimated_cost_usd"]
    sonnet_cost = tracker2.summary()["estimated_cost_usd"]
    assert opus_cost > sonnet_cost, \
        f"Opus ({opus_cost}) should cost more than Sonnet ({sonnet_cost})"
    print("PASS: test_cost_opus_vs_sonnet")


def test_unknown_model_uses_default():
    """Unknown model should use default pricing."""
    from utils.token_tracker import TokenTracker, _match_pricing, MODEL_PRICING

    pricing = _match_pricing("some-unknown-model-v99")
    assert pricing == MODEL_PRICING["_default"]

    tracker = TokenTracker()
    tracker.record("test", "unknown-model", {
        "prompt_tokens": 1000, "completion_tokens": 500,
    })
    s = tracker.summary()
    assert s["estimated_cost_usd"] > 0  # Should still compute cost
    print("PASS: test_unknown_model_uses_default")


def test_fuzzy_model_matching():
    """Fuzzy matching should handle model name variants."""
    from utils.token_tracker import _match_pricing, MODEL_PRICING

    # "claude-sonnet-4-5" is a substring of longer model IDs
    pricing = _match_pricing("claude-sonnet-4-5-20250929")
    assert pricing == MODEL_PRICING["claude-sonnet-4-5"]
    print("PASS: test_fuzzy_model_matching")


def test_none_usage_values():
    """Handle None values in usage dict gracefully."""
    from utils.token_tracker import TokenTracker

    tracker = TokenTracker()
    tracker.record("test", "claude-sonnet-4-5", {
        "prompt_tokens": None,
        "completion_tokens": None,
    })

    s = tracker.summary()
    assert s["total_tokens"] == 0
    assert s["total_calls"] == 1
    print("PASS: test_none_usage_values")


def test_missing_usage_keys():
    """Handle missing keys in usage dict gracefully."""
    from utils.token_tracker import TokenTracker

    tracker = TokenTracker()
    tracker.record("test", "claude-sonnet-4-5", {})

    s = tracker.summary()
    assert s["total_tokens"] == 0
    assert s["total_calls"] == 1
    print("PASS: test_missing_usage_keys")


def test_to_dict():
    """to_dict should include raw records and summary."""
    from utils.token_tracker import TokenTracker

    tracker = TokenTracker()
    tracker.record("discovery", "claude-sonnet-4-5", {
        "prompt_tokens": 100, "completion_tokens": 50,
    })

    d = tracker.to_dict()
    assert "records" in d
    assert "summary" in d
    assert len(d["records"]) == 1
    assert d["records"][0]["agent"] == "discovery"
    assert d["summary"]["total_tokens"] == 150
    print("PASS: test_to_dict")


def test_format_cli_summary():
    """CLI summary should be a readable string."""
    from utils.token_tracker import TokenTracker

    tracker = TokenTracker()
    tracker.record("discovery", "claude-sonnet-4-5", {
        "prompt_tokens": 500, "completion_tokens": 200,
    })
    tracker.record("synthesis", "claude-sonnet-4-5", {
        "prompt_tokens": 800, "completion_tokens": 400,
    })

    output = tracker.format_cli_summary()
    assert isinstance(output, str)
    assert "1,900" in output or "1900" in output  # total tokens
    assert "discovery" in output
    assert "synthesis" in output
    assert "$" in output  # cost
    print("PASS: test_format_cli_summary")


def test_elapsed_time():
    """Elapsed seconds should reflect real time."""
    from utils.token_tracker import TokenTracker

    tracker = TokenTracker()
    time.sleep(0.1)
    tracker.record("test", "claude-sonnet-4-5", {
        "prompt_tokens": 10, "completion_tokens": 5,
    })

    s = tracker.summary()
    assert s["elapsed_seconds"] >= 0.1
    print("PASS: test_elapsed_time")


def main():
    tests = [
        test_init,
        test_single_record,
        test_multiple_agents,
        test_multiple_models,
        test_cost_estimation,
        test_cost_opus_vs_sonnet,
        test_unknown_model_uses_default,
        test_fuzzy_model_matching,
        test_none_usage_values,
        test_missing_usage_keys,
        test_to_dict,
        test_format_cli_summary,
        test_elapsed_time,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

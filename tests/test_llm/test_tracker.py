def test_tracker_record_and_summary():
    from litscribe.llm.tracker import TokenTracker
    tracker = TokenTracker()
    tracker.record("discovery", "openai/qwen-turbo", {"prompt_tokens": 100, "completion_tokens": 50})
    tracker.record("synthesis", "openai/qwen-max", {"prompt_tokens": 500, "completion_tokens": 300})
    summary = tracker.summary()
    assert summary["total_prompt_tokens"] == 600
    assert summary["total_completion_tokens"] == 350
    assert "discovery" in summary["by_agent"]

def test_tracker_empty_summary():
    from litscribe.llm.tracker import TokenTracker
    tracker = TokenTracker()
    summary = tracker.summary()
    assert summary["total_prompt_tokens"] == 0

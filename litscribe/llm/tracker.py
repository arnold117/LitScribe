from __future__ import annotations
from collections import defaultdict

class TokenTracker:
    def __init__(self):
        self._records: list[dict] = []

    def record(self, agent_name: str, model: str, usage: dict):
        self._records.append({
            "agent": agent_name, "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        })

    def summary(self) -> dict:
        total_prompt = sum(r["prompt_tokens"] for r in self._records)
        total_completion = sum(r["completion_tokens"] for r in self._records)
        by_agent: dict = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
        for r in self._records:
            by_agent[r["agent"]]["prompt_tokens"] += r["prompt_tokens"]
            by_agent[r["agent"]]["completion_tokens"] += r["completion_tokens"]
            by_agent[r["agent"]]["calls"] += 1
        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_calls": len(self._records),
            "by_agent": dict(by_agent),
        }

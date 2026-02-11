"""Token usage tracking for LLM cost analysis.

Accumulates token usage across all LLM calls in a pipeline run,
providing per-agent and per-model breakdowns with cost estimation.

The tracker is stored in a ContextVar (not in LangGraph state) to avoid
msgpack serialization issues with LangGraph checkpointing.
"""

import contextvars
import time
from typing import Any, Dict, List, Optional


# ContextVar for the current tracker instance — avoids storing in LangGraph state
_tracker_var: contextvars.ContextVar["TokenTracker | None"] = contextvars.ContextVar(
    "token_tracker", default=None
)


def get_tracker() -> "TokenTracker | None":
    """Get the current TokenTracker from context."""
    return _tracker_var.get()


def set_tracker(tracker: "TokenTracker | None") -> contextvars.Token:
    """Set the current TokenTracker in context. Returns a reset token."""
    return _tracker_var.set(tracker)


# Pricing per 1M tokens (USD) — updated Feb 2026
MODEL_PRICING = {
    # Anthropic
    "claude-opus-4-5": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-3-5": {"input": 0.80, "output": 4.0},
    # DeepSeek
    "deepseek/deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
    # Qwen (DashScope) — prices in USD (converted from RMB at ~7.2)
    "dashscope/qwen3-max": {"input": 0.35, "output": 1.39},
    "dashscope/qwen-max": {"input": 2.40, "output": 9.60},
    "dashscope/qwen-plus": {"input": 0.80, "output": 2.00},
    "dashscope/qwen-turbo": {"input": 0.30, "output": 0.60},
    # Fallback
    "_default": {"input": 3.0, "output": 15.0},
}


def _match_pricing(model: str) -> Dict[str, float]:
    """Find pricing for a model, with fuzzy matching."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Fuzzy: check if any key is a substring of model
    for key, pricing in MODEL_PRICING.items():
        if key != "_default" and key in model:
            return pricing
    return MODEL_PRICING["_default"]


class TokenTracker:
    """Accumulates token usage across LLM calls in a pipeline run."""

    def __init__(self):
        self._records: List[Dict[str, Any]] = []
        self._start_time: float = time.time()

    def record(self, agent: str, model: str, usage: Dict[str, Any]) -> None:
        """Record a single LLM call's token usage.

        Args:
            agent: Agent name (e.g. "discovery", "synthesis")
            model: Model identifier (e.g. "claude-sonnet-4-5")
            usage: Dict with prompt_tokens and completion_tokens
        """
        self._records.append({
            "agent": agent,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "timestamp": time.time(),
        })

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of all recorded token usage.

        Returns:
            Dict with total tokens, per-agent breakdown, per-model breakdown,
            and estimated cost in USD.
        """
        total_prompt = 0
        total_completion = 0
        by_agent: Dict[str, Dict[str, int]] = {}
        by_model: Dict[str, Dict[str, int]] = {}

        for rec in self._records:
            pt = rec["prompt_tokens"]
            ct = rec["completion_tokens"]
            total_prompt += pt
            total_completion += ct

            agent = rec["agent"]
            if agent not in by_agent:
                by_agent[agent] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
            by_agent[agent]["prompt_tokens"] += pt
            by_agent[agent]["completion_tokens"] += ct
            by_agent[agent]["calls"] += 1

            model = rec["model"]
            if model not in by_model:
                by_model[model] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
            by_model[model]["prompt_tokens"] += pt
            by_model[model]["completion_tokens"] += ct
            by_model[model]["calls"] += 1

        # Cost estimation
        total_cost = 0.0
        model_costs: Dict[str, float] = {}
        for model, counts in by_model.items():
            pricing = _match_pricing(model)
            cost = (
                counts["prompt_tokens"] / 1_000_000 * pricing["input"]
                + counts["completion_tokens"] / 1_000_000 * pricing["output"]
            )
            model_costs[model] = round(cost, 4)
            total_cost += cost

        elapsed = time.time() - self._start_time

        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_calls": len(self._records),
            "estimated_cost_usd": round(total_cost, 4),
            "by_agent": by_agent,
            "by_model": by_model,
            "model_costs_usd": model_costs,
            "elapsed_seconds": round(elapsed, 1),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization including individual records."""
        return {
            "records": self._records,
            "summary": self.summary(),
        }

    def format_cli_summary(self) -> str:
        """Format a human-readable summary for CLI output."""
        s = self.summary()
        lines = []
        total = s["total_tokens"]
        prompt = s["total_prompt_tokens"]
        completion = s["total_completion_tokens"]
        cost = s["estimated_cost_usd"]

        lines.append(f"Tokens: {prompt:,} prompt + {completion:,} completion = {total:,} total")
        lines.append(f"Cost: ${cost:.4f} ({s['total_calls']} LLM calls, {s['elapsed_seconds']}s)")

        # Per-model costs
        if s["model_costs_usd"]:
            parts = []
            for model, mc in sorted(s["model_costs_usd"].items(), key=lambda x: -x[1]):
                short_name = model.split("/")[-1] if "/" in model else model
                parts.append(f"{short_name}: ${mc:.4f}")
            lines.append(f"  By model: {', '.join(parts)}")

        # Per-agent breakdown (top 5 by token usage)
        if s["by_agent"]:
            agent_totals = {
                a: d["prompt_tokens"] + d["completion_tokens"]
                for a, d in s["by_agent"].items()
            }
            sorted_agents = sorted(agent_totals.items(), key=lambda x: -x[1])[:5]
            if total > 0:
                parts = [f"{a} {t*100//total}%" for a, t in sorted_agents]
            else:
                parts = [f"{a} 0%" for a, _ in sorted_agents]
            lines.append(f"  By agent: {', '.join(parts)}")

        return "\n".join(lines)

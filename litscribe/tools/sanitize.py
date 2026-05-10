from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
    r"you\s+are\s+now\s+a",
    r"system\s*:\s*",
    r"<\|?(system|assistant|user)\|?>",
    r"forget\s+(everything|all|your)",
    r"override\s+(your|the)\s+(instructions|rules)",
    r"pretend\s+(you|to)\s+",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def sanitize_input(text: str, max_length: int = 500) -> str:
    text = text[:max_length]

    for pattern in _compiled:
        if pattern.search(text):
            logger.warning(f"Prompt injection attempt detected: {text[:50]}")
            text = pattern.sub("", text)

    return text.strip()


def sanitize_research_question(question: str) -> str:
    return sanitize_input(question, max_length=300)

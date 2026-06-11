"""Wire the writing-template library (prompts/templates.py) into runnable actions.

Built-in templates plus user custom templates are filled with the current
review's papers + user instructions and sent to the model. Custom templates
are persisted as JSON under the litscribe data directory.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from litscribe.prompts.templates import TEMPLATES
from litscribe.tools.cite_keys import assign_cite_keys

logger = logging.getLogger(__name__)

# Human-facing labels (bilingual) for the built-in templates.
TEMPLATE_LABELS = {
    "related-work": "Related Work · 相关工作",
    "grant-background": "Grant Background · 基金研究背景",
    "research-proposal": "Research Proposal · 开题报告",
    "abstract-generate": "Abstract · 生成摘要",
    "abstract-rewrite": "Abstract Rewrite · 润色摘要",
    "translation": "Translation · 翻译润色",
    "rebuttal": "Rebuttal · 审稿回复",
}


def _custom_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "custom_templates.json"


def load_custom(data_dir: str | Path) -> dict:
    p = _custom_path(data_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read custom templates: {e}")
    return {}


def save_custom(data_dir: str | Path, templates: dict) -> None:
    p = _custom_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(templates, ensure_ascii=False, indent=2), encoding="utf-8")


# Built-ins that genuinely need a review's papers to cite. translation /
# abstract-rewrite operate on the user's own pasted text, so they don't.
_BUILTIN_NEEDS_PAPERS = {
    "related-work", "grant-background", "research-proposal",
    "abstract-generate", "rebuttal",
}


def _needs_papers(prompt: str) -> bool:
    return "{papers}" in prompt or "{citation_checklist}" in prompt


def list_templates(data_dir: str | Path) -> list[dict]:
    items = [
        {
            "id": tid,
            "label": TEMPLATE_LABELS.get(tid, tid),
            "builtin": True,
            "needs_papers": tid in _BUILTIN_NEEDS_PAPERS,
        }
        for tid, tpl in TEMPLATES.items()
    ]
    for tid, t in load_custom(data_dir).items():
        items.append({
            "id": tid,
            "label": t.get("label", tid),
            "builtin": False,
            "needs_papers": _needs_papers(t.get("prompt", "")),
        })
    return items


def _format_papers(papers, key_map: dict) -> str:
    lines = []
    for p in papers:
        key = key_map.get(p.paper_id, "?")
        authors = ", ".join(p.authors[:3]) if p.authors else "Unknown"
        if p.authors and len(p.authors) > 3:
            authors += " et al."
        abstract = (p.abstract or "").strip().replace("\n", " ")[:300]
        lines.append(f"[@{key}] {authors} ({p.year}). {p.title}.\n  {abstract}")
    return "\n\n".join(lines)


def _checklist(papers, key_map: dict) -> str:
    return "\n".join(
        f"- [@{key_map.get(p.paper_id, '?')}] {p.title[:60]}" for p in papers
    )


async def apply_template(
    model,
    template_id: str,
    papers: list,
    user_instructions: str,
    word_count: int = 800,
    data_dir: str | Path | None = None,
) -> str:
    """Fill a template with the current papers + instructions and run it."""
    tpl = TEMPLATES.get(template_id)
    if tpl is None and data_dir is not None:
        custom = load_custom(data_dir).get(template_id)
        tpl = custom.get("prompt") if custom else None
    if not tpl:
        raise ValueError(f"Unknown template: {template_id}")

    key_map = assign_cite_keys(papers) if papers else {}
    fields = {
        "user_instructions": user_instructions or "(none provided)",
        "papers": _format_papers(papers, key_map) if papers else "(no papers available)",
        "num_papers": str(len(papers)),
        "citation_checklist": _checklist(papers, key_map) if papers else "(none)",
        "word_count": str(word_count),
    }
    needed = set(re.findall(r"\{(\w+)\}", tpl))
    safe = {k: fields.get(k, "") for k in needed}

    result = await model.ainvoke(tpl.format(**safe))
    return result.content.strip()

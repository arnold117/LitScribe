"""Procedural memory — filesystem skills with vector search retrieval."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from litscribe.store.vectors import VectorStore

SKILL_COLLECTION = "skill_embeddings"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_VERSION_RE = re.compile(r"^(version:\s*)(\d+)", re.MULTILINE)
_STRATEGY_SECTION_RE = re.compile(r"(## Strategy\n)(.*?)(\n## |\Z)", re.DOTALL)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _build_skill_content(
    name: str,
    domain: str,
    version: int,
    success_rate: float,
    last_used: str,
    trigger: str,
    strategy: str,
    learned_adjustments: list[str],
) -> str:
    adj_block = "\n".join(f"- {a}" for a in learned_adjustments) if learned_adjustments else ""
    return (
        f"---\n"
        f"name: {name}\n"
        f"domain: {domain}\n"
        f"version: {version}\n"
        f"success_rate: {success_rate}\n"
        f"last_used: {last_used}\n"
        f"---\n"
        f"## Trigger\n{trigger}\n\n"
        f"## Strategy\n{strategy}\n\n"
        f"## Learned Adjustments\n{adj_block}\n"
    )


def _parse_skill(path: Path) -> dict[str, Any] | None:
    """Parse a skill Markdown file with YAML frontmatter."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None

    fm_match = _FRONTMATTER_RE.match(raw)
    if not fm_match:
        return None

    # Parse simple key: value frontmatter (no nested structures)
    fm_text = fm_match.group(1)
    meta: dict[str, Any] = {}
    for line in fm_text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    body = raw[fm_match.end():]

    # Extract trigger
    trigger = ""
    trigger_match = re.search(r"## Trigger\n(.*?)(?:\n## |\Z)", body, re.DOTALL)
    if trigger_match:
        trigger = trigger_match.group(1).strip()

    # Extract strategy
    strategy = ""
    strategy_match = re.search(r"## Strategy\n(.*?)(?:\n## |\Z)", body, re.DOTALL)
    if strategy_match:
        strategy = strategy_match.group(1).strip()

    return {
        "slug": path.stem,
        "name": meta.get("name", path.stem),
        "domain": meta.get("domain", ""),
        "version": int(meta.get("version", 1)),
        "success_rate": float(meta.get("success_rate", 0.0)),
        "last_used": meta.get("last_used", ""),
        "trigger": trigger,
        "strategy": strategy,
        "raw_content": raw,
    }


class ProceduralMemory:
    """Stores and retrieves research skills as versioned Markdown files."""

    def __init__(self, skills_dir: Path, vectors: VectorStore):
        self._dir = Path(skills_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors = vectors

    def _path(self, slug: str) -> Path:
        return self._dir / f"{slug}.md"

    def save_skill(
        self,
        name: str,
        domain: str,
        trigger: str,
        strategy: str,
        learned_adjustments: list[str],
        success_rate: float = 0.0,
    ) -> str:
        """Persist a new skill file and index it for vector search. Returns slug."""
        slug = _slugify(name)
        last_used = datetime.now(timezone.utc).isoformat()
        content = _build_skill_content(
            name=name,
            domain=domain,
            version=1,
            success_rate=success_rate,
            last_used=last_used,
            trigger=trigger,
            strategy=strategy,
            learned_adjustments=learned_adjustments,
        )
        self._path(slug).write_text(content, encoding="utf-8")
        self._index_skill(slug, name, domain, trigger, strategy)
        return slug

    def _index_skill(self, slug: str, name: str, domain: str, trigger: str, strategy: str) -> None:
        """Upsert skill in vector store for semantic retrieval."""
        text = f"{name}. Domain: {domain}. Trigger: {trigger}. Strategy: {strategy}"
        # Delete existing entry first to allow upsert semantics
        self._vectors.delete(SKILL_COLLECTION, [slug])
        self._vectors.add_texts(
            SKILL_COLLECTION,
            texts=[text],
            metadatas=[{"slug": slug, "name": name, "domain": domain}],
            ids=[slug],
        )

    def get_skill(self, slug: str) -> dict[str, Any] | None:
        """Load and parse a skill by slug, or None if not found."""
        return _parse_skill(self._path(slug))

    def list_skills(self) -> list[dict[str, Any]]:
        """Return all skills in the skills directory."""
        skills = []
        for p in sorted(self._dir.glob("*.md")):
            parsed = _parse_skill(p)
            if parsed is not None:
                skills.append(parsed)
        return skills

    def patch_skill(
        self,
        slug: str,
        strategy: str | None = None,
        adjustment: str | None = None,
        score: float | None = None,
    ) -> bool:
        """Increment version, optionally update strategy, append an adjustment,
        and update success_rate via exponential moving average."""
        path = self._path(slug)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")

        # Increment version in frontmatter
        def _bump(m: re.Match) -> str:
            return f"{m.group(1)}{int(m.group(2)) + 1}"

        content = _VERSION_RE.sub(_bump, content, count=1)

        # Update success_rate via EMA: new = 0.7 * old + 0.3 * score
        if score is not None:
            def _update_rate(m: re.Match) -> str:
                old_rate = float(m.group(2))
                new_rate = round(0.7 * old_rate + 0.3 * score, 3)
                return f"{m.group(1)}{new_rate}"
            content = re.sub(
                r"^(success_rate:\s*)([\d.]+)",
                _update_rate,
                content,
                count=1,
                flags=re.MULTILINE,
            )

        # Replace strategy section if provided
        if strategy is not None:
            content = re.sub(
                r"(## Strategy\n)(.*?)(\n## |\Z)",
                lambda m: f"{m.group(1)}{strategy}\n{m.group(3)}",
                content,
                count=1,
                flags=re.DOTALL,
            )

        # Append adjustment under Learned Adjustments
        if adjustment is not None:
            adj_header = "## Learned Adjustments\n"
            if adj_header in content:
                content = content.replace(adj_header, f"{adj_header}- {adjustment}\n", 1)
            else:
                content += f"\n{adj_header}- {adjustment}\n"

        path.write_text(content, encoding="utf-8")

        # Re-index with updated text
        parsed = _parse_skill(path)
        if parsed:
            self._index_skill(slug, parsed["name"], parsed["domain"], parsed["trigger"], parsed["strategy"])
        return True

    def find_relevant(self, query: str, n: int = 5) -> list[dict[str, Any]]:
        """Semantic search for skills relevant to a query."""
        hits = self._vectors.search(query, SKILL_COLLECTION, n)
        results = []
        for hit in hits:
            slug = hit["metadata"].get("slug", hit["id"])
            skill = self.get_skill(slug)
            if skill is not None:
                results.append(skill)
        return results

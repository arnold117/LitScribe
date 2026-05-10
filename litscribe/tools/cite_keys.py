from __future__ import annotations

import re
from typing import Any

from litscribe.models.paper import Paper


def _extract_last_name(author: str) -> str:
    author = author.strip()
    if not author:
        return "unknown"
    if "," in author:
        return author.split(",")[0].strip().lower()
    parts = author.split()
    if not parts:
        return "unknown"
    last = parts[-1].rstrip(".")
    if last.isupper() and len(last) <= 3 and len(parts) > 1:
        return parts[0].strip().lower()
    return last.lower()


def generate_cite_key(paper: Paper, existing_keys: set[str]) -> str:
    name = "unknown"
    if paper.authors:
        name = _extract_last_name(paper.authors[0])
        name = re.sub(r"[^a-z]", "", name)
        if not name:
            name = "unknown"

    year = str(paper.year) if paper.year else "nd"
    base = f"{name}{year}"

    key = base
    suffix = ord("a")
    while key in existing_keys:
        key = f"{base}{chr(suffix)}"
        suffix += 1

    return key


def assign_cite_keys(papers: list[Paper]) -> dict[str, str]:
    existing: set[str] = set()
    mapping: dict[str, str] = {}

    for paper in papers:
        key = generate_cite_key(paper, existing)
        existing.add(key)
        mapping[paper.paper_id] = key

    return mapping


def build_cite_key_table(papers: list[Paper], key_map: dict[str, str]) -> str:
    lines = []
    for p in papers:
        key = key_map.get(p.paper_id, "unknown")
        authors = ", ".join(p.authors[:2]) if p.authors else "Unknown"
        if len(p.authors) > 2:
            authors += " et al."
        lines.append(f"- `@{key}` → {authors} ({p.year}). {p.title[:80]}")
    return "\n".join(lines)


def build_bibtex(papers: list[Paper], key_map: dict[str, str]) -> str:
    entries = []
    for p in papers:
        key = key_map.get(p.paper_id, "unknown")
        authors = " and ".join(p.authors[:5]) if p.authors else "Unknown"
        venue = p.venue or ""

        entry_type = "article"
        if venue and any(k in venue.lower() for k in ("conference", "proc", "workshop", "icml", "neurips", "aaai")):
            entry_type = "inproceedings"
        elif "arxiv" in p.paper_id.lower():
            entry_type = "misc"

        fields = [
            f"  author = {{{authors}}}",
            f"  title = {{{p.title}}}",
            f"  year = {{{p.year or 'n.d.'}}}",
        ]
        if venue:
            field_name = "booktitle" if entry_type == "inproceedings" else "journal"
            fields.append(f"  {field_name} = {{{venue}}}")
        if p.doi:
            fields.append(f"  doi = {{{p.doi}}}")

        entry = f"@{entry_type}{{{key},\n" + ",\n".join(fields) + "\n}"
        entries.append(entry)

    return "\n\n".join(entries)

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OutlineNode:
    title: str
    level: int
    children: list[OutlineNode] = field(default_factory=list)
    number: str = ""

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leaves(self) -> list[OutlineNode]:
        if self.is_leaf:
            return [self]
        result = []
        for c in self.children:
            result.extend(c.leaves())
        return result

    def full_title(self, ancestors: list[str] | None = None) -> str:
        parts = list(ancestors or [])
        parts.append(self.title)
        return " > ".join(parts)


_HEADING_PATTERN = re.compile(
    r"^(?:第[一二三四五六七八九十百千\d]+[章节篇]|"
    r"(\d+(?:\.\d+)*))\s*(.+)",
)
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)")


def _detect_level(number: str) -> int:
    if not number:
        return 0
    return number.count(".") + 1


def parse_outline_text(lines: list[str]) -> list[OutlineNode]:
    nodes: list[OutlineNode] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        md_m = _MD_HEADING.match(line)
        if md_m:
            level = len(md_m.group(1))
            title = md_m.group(2).strip()
            nodes.append(OutlineNode(title=title, level=level))
            continue

        m = _HEADING_PATTERN.match(line)
        if m:
            number = m.group(1) or ""
            title = m.group(2).strip() if m.group(2) else line
            level = _detect_level(number)
            if not number and line.startswith("第"):
                level = 0
            nodes.append(OutlineNode(title=title, level=level, number=number))
            continue

        if line in ("引言", "本章小结", "小结", "总结", "结论", "Introduction", "Conclusion", "Summary"):
            continue

        nodes.append(OutlineNode(title=line, level=0))

    return _build_tree(nodes)


def _build_tree(flat: list[OutlineNode]) -> list[OutlineNode]:
    if not flat:
        return []

    roots: list[OutlineNode] = []
    stack: list[OutlineNode] = []

    for node in flat:
        while stack and stack[-1].level >= node.level:
            stack.pop()

        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    return roots


def parse_docx_outline(path: str | Path) -> list[OutlineNode]:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return parse_outline_text(lines)


def parse_markdown_outline(path: str | Path) -> list[OutlineNode]:
    text = Path(path).read_text(encoding="utf-8")
    return parse_outline_text(text.splitlines())


def parse_outline(path: str | Path) -> list[OutlineNode]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".docx":
        return parse_docx_outline(p)
    elif suffix in (".md", ".markdown", ".txt"):
        return parse_markdown_outline(p)
    else:
        raise ValueError(f"Unsupported outline format: {suffix}. Use .docx, .md, or .txt")


def outline_to_sections(roots: list[OutlineNode]) -> list[dict]:
    sections = []

    def _walk(node: OutlineNode, ancestors: list[str]):
        path = ancestors + [node.title]
        if node.is_leaf:
            sections.append({
                "title": node.title,
                "number": node.number,
                "level": node.level,
                "path": path,
                "query": node.title,
            })
        else:
            for child in node.children:
                _walk(child, path)

    for root in roots:
        _walk(root, [])

    return sections

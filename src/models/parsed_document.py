"""Models for parsed PDF documents."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Section:
    """Represents a section in a parsed document."""

    title: str
    content: str
    level: int  # Heading level (1, 2, 3, etc.)
    start_page: int
    end_page: int

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "start_page": self.start_page,
            "end_page": self.end_page,
        }


@dataclass
class Table:
    """Represents a table extracted from a PDF."""

    caption: str
    content: str  # Markdown format
    page_num: int
    table_id: str = ""  # e.g., "Table 1"

    def to_dict(self) -> dict:
        return {
            "caption": self.caption,
            "content": self.content,
            "page_num": self.page_num,
            "table_id": self.table_id,
        }


@dataclass
class Equation:
    """Represents a mathematical equation."""

    latex: str  # LaTeX representation
    page_num: int
    context: str = ""  # Surrounding text
    equation_id: str = ""  # e.g., "Equation (1)"

    def to_dict(self) -> dict:
        return {
            "latex": self.latex,
            "page_num": self.page_num,
            "context": self.context,
            "equation_id": self.equation_id,
        }


@dataclass
class Citation:
    """Represents a citation in the document."""

    text: str  # The citation text
    page_num: int
    context: str  # Surrounding text
    ref_id: str = ""  # e.g., "[1]", "[Smith2024]"
    reference: str = ""  # Full reference from bibliography

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "page_num": self.page_num,
            "context": self.context,
            "ref_id": self.ref_id,
            "reference": self.reference,
        }


@dataclass
class ParsedDocument:
    """Represents a fully parsed academic PDF document."""

    markdown: str  # Full document in markdown
    sections: List[Section] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    references: List[str] = field(default_factory=list)  # Bibliography
    metadata: Dict[str, Any] = field(default_factory=dict)  # Pages, word count, etc.

    def to_dict(self) -> dict:
        return {
            "markdown": self.markdown,
            "sections": [s.to_dict() for s in self.sections],
            "tables": [t.to_dict() for t in self.tables],
            "equations": [e.to_dict() for e in self.equations],
            "citations": [c.to_dict() for c in self.citations],
            "references": self.references,
            "metadata": self.metadata,
        }

    @property
    def num_pages(self) -> int:
        """Get number of pages."""
        return self.metadata.get("num_pages", 0)

    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return self.metadata.get("word_count", len(self.markdown.split()))

    def get_section(self, title: str) -> Section | None:
        """Get a section by title (case-insensitive)."""
        title_lower = title.lower()
        for section in self.sections:
            if title_lower in section.title.lower():
                return section
        return None

    def get_abstract(self) -> str:
        """Extract abstract section."""
        abstract_section = self.get_section("abstract")
        if abstract_section:
            return abstract_section.content
        return ""

    def get_conclusion(self) -> str:
        """Extract conclusion section."""
        conclusion_section = self.get_section("conclusion")
        if conclusion_section:
            return conclusion_section.content
        return ""

    def __repr__(self) -> str:
        return (
            f"ParsedDocument("
            f"pages={self.num_pages}, "
            f"sections={len(self.sections)}, "
            f"tables={len(self.tables)}, "
            f"equations={len(self.equations)}, "
            f"citations={len(self.citations)})"
        )

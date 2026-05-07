"""Pandoc exporter for LitScribe v2.

Converts review text to various document formats using Pandoc:
- Word (.docx)
- PDF (via LaTeX)
- HTML
- EPUB
- Markdown

Requires pandoc to be installed: https://pandoc.org/installing.html
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from litscribe.exporters.bibtex import papers_to_bibtex
from litscribe.exporters.citation_formatter import CitationStyle
from litscribe.models.paper import Paper


class ExportFormat(Enum):
    """Supported export formats."""

    DOCX = "docx"
    PDF = "pdf"
    HTML = "html"
    LATEX = "latex"
    EPUB = "epub"
    MARKDOWN = "md"


@dataclass
class ExportConfig:
    """Configuration for export."""

    format: ExportFormat = ExportFormat.DOCX
    citation_style: CitationStyle = CitationStyle.APA
    language: str = "en"  # "en" or "zh"
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None

    # PDF/LaTeX specific
    template: Optional[str] = None
    fontsize: str = "11pt"
    geometry: str = "margin=1in"

    # Document options
    include_toc: bool = True
    include_references: bool = True
    number_sections: bool = True

    # Additional Pandoc args
    extra_args: List[str] = field(default_factory=list)


def export_review(
    text: str,
    papers: list[Paper],
    config: ExportConfig,
    output_path: str,
) -> str:
    """Export a review to the configured format via Pandoc.

    Writes markdown (with optional YAML front matter and references) to a
    temporary file, then calls pandoc to produce the final output.

    Args:
        text: The review body text (Markdown).
        papers: List of v2 Paper models to include as references.
        config: Export configuration.
        output_path: Destination file path (extension will be corrected).

    Returns:
        Absolute path to the exported file.

    Raises:
        RuntimeError: If Pandoc is not available or the conversion fails.
    """
    out = Path(output_path)
    ext = f".{config.format.value}"
    if out.suffix != ext:
        out = out.with_suffix(ext)

    markdown = _build_markdown(text, papers, config)

    # Markdown — write directly, no pandoc required
    if config.format == ExportFormat.MARKDOWN:
        out.write_text(markdown, encoding="utf-8")
        return str(out)

    if not shutil.which("pandoc"):
        raise RuntimeError(
            "Pandoc is not installed. Please install from https://pandoc.org/installing.html"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        md_path = tmp / "review.md"
        md_path.write_text(markdown, encoding="utf-8")

        bib_path: Optional[Path] = None
        if config.include_references and papers:
            bib_path = tmp / "references.bib"
            bib_path.write_text(papers_to_bibtex(papers), encoding="utf-8")

        cmd = ["pandoc", str(md_path), "-o", str(out)]

        if bib_path:
            cmd.extend(["--bibliography", str(bib_path), "--citeproc"])

        if config.format == ExportFormat.PDF:
            cmd.append("--pdf-engine=xelatex")
            if config.language == "zh":
                cmd.extend(["-V", "CJKmainfont=PingFang SC"])

        if config.format == ExportFormat.DOCX and config.template:
            cmd.extend(["--reference-doc", config.template])

        if config.format == ExportFormat.HTML:
            cmd.extend(["--standalone", "--self-contained"])

        cmd.extend(config.extra_args)

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Pandoc export failed: {e.stderr}")

    return str(out)


def _build_markdown(
    text: str,
    papers: list[Paper],
    config: ExportConfig,
) -> str:
    """Build the full Markdown document with YAML front matter and references."""
    from datetime import datetime
    from litscribe.exporters.citation_formatter import format_citations

    lines: list[str] = []

    # YAML front matter
    lines.append("---")
    if config.title:
        lines.append(f'title: "{config.title}"')
    if config.author:
        lines.append(f'author: "{config.author}"')
    date = config.date or datetime.now().strftime("%Y-%m-%d")
    lines.append(f'date: "{date}"')
    if config.language == "zh":
        lines.append("lang: zh-CN")
        lines.append('CJKmainfont: "PingFang SC"')
    else:
        lines.append("lang: en")
    if config.format in (ExportFormat.PDF, ExportFormat.LATEX):
        lines.append(f"fontsize: {config.fontsize}")
        lines.append(f"geometry: {config.geometry}")
    if config.include_toc:
        lines.append("toc: true")
        lines.append("toc-depth: 3")
    if config.number_sections:
        lines.append("numbersections: true")
    lines.append("---")
    lines.append("")

    # Main body
    if text:
        lines.append(text)
        lines.append("")

    # References section
    if config.include_references and papers:
        heading = "# References" if config.language == "en" else "# 参考文献"
        lines.append(heading)
        lines.append("")
        numbered_styles = {CitationStyle.IEEE, CitationStyle.GB_T_7714}
        citations = format_citations(papers, config.citation_style)
        for citation in citations:
            lines.append(citation)
            lines.append("")

    return "\n".join(lines)


__all__ = [
    "ExportFormat",
    "ExportConfig",
    "export_review",
]

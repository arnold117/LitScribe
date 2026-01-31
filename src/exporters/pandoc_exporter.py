"""Pandoc exporter for LitScribe.

Converts literature reviews to various formats using Pandoc:
- Word (.docx)
- PDF (via LaTeX)
- HTML
- EPUB

Requires pandoc to be installed: https://pandoc.org/installing.html
"""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.state import LitScribeState, PaperSummary, SynthesisOutput
from exporters.bibtex_exporter import BibTeXExporter
from exporters.citation_formatter import CitationFormatter, CitationStyle


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
    template: Optional[str] = None  # Template name or path
    fontsize: str = "11pt"
    geometry: str = "margin=1in"

    # Document options
    include_abstract: bool = True
    include_toc: bool = True
    include_references: bool = True
    number_sections: bool = True

    # Additional Pandoc args
    extra_args: List[str] = field(default_factory=list)


class PandocExporter:
    """Export literature reviews using Pandoc."""

    def __init__(
        self,
        state: LitScribeState,
        config: Optional[ExportConfig] = None,
    ):
        """Initialize exporter.

        Args:
            state: LitScribe workflow state
            config: Export configuration
        """
        self.state = state
        self.config = config or ExportConfig()
        self.papers = state.get("analyzed_papers", [])
        self.synthesis = state.get("synthesis")

        # Check Pandoc availability
        self._pandoc_available = self._check_pandoc()

    def _check_pandoc(self) -> bool:
        """Check if Pandoc is installed."""
        return shutil.which("pandoc") is not None

    def _generate_markdown(self) -> str:
        """Generate Markdown content from state.

        Returns:
            Markdown string with YAML front matter
        """
        lines = []

        # YAML front matter
        lines.append("---")

        # Title
        title = self.config.title or f"Literature Review: {self.state.get('research_question', 'Untitled')}"
        lines.append(f'title: "{title}"')

        # Author
        if self.config.author:
            lines.append(f'author: "{self.config.author}"')

        # Date
        if self.config.date:
            lines.append(f'date: "{self.config.date}"')
        else:
            from datetime import datetime
            lines.append(f'date: "{datetime.now().strftime("%Y-%m-%d")}"')

        # Language
        if self.config.language == "zh":
            lines.append('lang: zh-CN')
            lines.append('CJKmainfont: "PingFang SC"')  # macOS Chinese font
        else:
            lines.append('lang: en')

        # PDF options
        if self.config.format in (ExportFormat.PDF, ExportFormat.LATEX):
            lines.append(f'fontsize: {self.config.fontsize}')
            lines.append(f'geometry: {self.config.geometry}')

        # TOC
        if self.config.include_toc:
            lines.append('toc: true')
            lines.append('toc-depth: 3')

        # Section numbering
        if self.config.number_sections:
            lines.append('numbersections: true')

        lines.append("---")
        lines.append("")

        # Abstract
        if self.config.include_abstract and self.synthesis:
            abstract = self._generate_abstract()
            if abstract:
                lines.append("# Abstract")
                lines.append("")
                lines.append(abstract)
                lines.append("")

        # Main content from synthesis
        if self.synthesis:
            review_text = self.synthesis.get("review_text", "")
            if review_text:
                lines.append(review_text)
                lines.append("")

            # Research gaps
            gaps = self.synthesis.get("gaps", [])
            if gaps:
                lines.append("# Research Gaps" if self.config.language == "en" else "# 研究空白")
                lines.append("")
                for gap in gaps:
                    lines.append(f"- {gap}")
                lines.append("")

            # Future directions
            future = self.synthesis.get("future_directions", [])
            if future:
                lines.append("# Future Directions" if self.config.language == "en" else "# 未来方向")
                lines.append("")
                for direction in future:
                    lines.append(f"- {direction}")
                lines.append("")

        # References
        if self.config.include_references and self.papers:
            lines.append("# References" if self.config.language == "en" else "# 参考文献")
            lines.append("")

            formatter = CitationFormatter(self.config.citation_style)
            numbered = self.config.citation_style in (
                CitationStyle.IEEE,
                CitationStyle.GB_T_7714,
            )
            citations = formatter.format_papers(self.papers, numbered=numbered)

            for citation in citations:
                lines.append(f"{citation}")
                lines.append("")

        return "\n".join(lines)

    def _generate_abstract(self) -> str:
        """Generate abstract from synthesis.

        Returns:
            Abstract text
        """
        if not self.synthesis:
            return ""

        # Build abstract from themes
        themes = self.synthesis.get("themes", [])
        if not themes:
            return ""

        parts = []

        # Introduction sentence
        research_question = self.state.get("research_question", "")
        num_papers = len(self.papers)

        if self.config.language == "zh":
            parts.append(f'本文献综述分析了{num_papers}篇关于"{research_question}"的论文。')
        else:
            parts.append(f"This literature review analyzes {num_papers} papers on the topic of '{research_question}'.")

        # Theme summary
        if self.config.language == "zh":
            theme_names = [t.get("theme", "") for t in themes if t.get("theme")]
            if theme_names:
                parts.append(f"主要主题包括：{'、'.join(theme_names[:3])}等。")
        else:
            theme_names = [t.get("theme", "") for t in themes if t.get("theme")]
            if theme_names:
                parts.append(f"Key themes identified include: {', '.join(theme_names[:3])}.")

        # Gap mention
        gaps = self.synthesis.get("gaps", [])
        if gaps:
            if self.config.language == "zh":
                parts.append(f"研究发现了{len(gaps)}个研究空白需要未来工作关注。")
            else:
                parts.append(f"The review identifies {len(gaps)} research gaps for future work.")

        return " ".join(parts)

    def export(self, output_path: Path) -> Path:
        """Export to the configured format.

        Args:
            output_path: Path for the output file (extension will be corrected)

        Returns:
            Path to the exported file

        Raises:
            RuntimeError: If Pandoc is not available or export fails
        """
        output_path = Path(output_path)

        # Ensure correct extension
        ext = f".{self.config.format.value}"
        if output_path.suffix != ext:
            output_path = output_path.with_suffix(ext)

        # If just Markdown, write directly
        if self.config.format == ExportFormat.MARKDOWN:
            content = self._generate_markdown()
            output_path.write_text(content, encoding="utf-8")
            return output_path

        # Check Pandoc
        if not self._pandoc_available:
            raise RuntimeError(
                "Pandoc is not installed. Please install from https://pandoc.org/installing.html"
            )

        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write Markdown source
            md_path = tmpdir / "review.md"
            md_content = self._generate_markdown()
            md_path.write_text(md_content, encoding="utf-8")

            # Generate BibTeX if needed
            bib_path = None
            if self.config.include_references and self.papers:
                bib_path = tmpdir / "references.bib"
                exporter = BibTeXExporter(self.papers)
                exporter.save(bib_path)

            # Build Pandoc command
            cmd = ["pandoc", str(md_path)]

            # Output format
            cmd.extend(["-o", str(output_path)])

            # Add BibTeX bibliography
            if bib_path:
                cmd.extend(["--bibliography", str(bib_path)])
                cmd.append("--citeproc")

            # Format-specific options
            if self.config.format == ExportFormat.PDF:
                cmd.append("--pdf-engine=xelatex")  # Better Unicode support
                if self.config.language == "zh":
                    cmd.append("-V")
                    cmd.append("CJKmainfont=PingFang SC")

            if self.config.format == ExportFormat.DOCX:
                # Use reference doc if provided
                if self.config.template:
                    cmd.extend(["--reference-doc", self.config.template])

            if self.config.format == ExportFormat.HTML:
                cmd.append("--standalone")
                cmd.append("--self-contained")

            # Add extra args
            cmd.extend(self.config.extra_args)

            # Run Pandoc
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Pandoc export failed: {e.stderr}")

        return output_path

    def export_markdown(self, output_path: Path) -> Path:
        """Export as Markdown (no Pandoc needed).

        Args:
            output_path: Path for the output file

        Returns:
            Path to the exported file
        """
        self.config.format = ExportFormat.MARKDOWN
        return self.export(output_path)


def export_review(
    state: LitScribeState,
    output_path: Path,
    format: ExportFormat = ExportFormat.DOCX,
    citation_style: CitationStyle = CitationStyle.APA,
    language: str = "en",
    **kwargs,
) -> Path:
    """Convenience function to export a literature review.

    Args:
        state: LitScribe workflow state
        output_path: Path for the output file
        format: Export format
        citation_style: Citation style for references
        language: "en" or "zh"
        **kwargs: Additional ExportConfig options

    Returns:
        Path to the exported file
    """
    config = ExportConfig(
        format=format,
        citation_style=citation_style,
        language=language,
        **kwargs,
    )
    exporter = PandocExporter(state, config)
    return exporter.export(output_path)


def generate_review_markdown(
    state: LitScribeState,
    citation_style: CitationStyle = CitationStyle.APA,
    language: str = "en",
) -> str:
    """Generate Markdown content from state (no Pandoc needed).

    Args:
        state: LitScribe workflow state
        citation_style: Citation style for references
        language: "en" or "zh"

    Returns:
        Markdown content string
    """
    config = ExportConfig(
        format=ExportFormat.MARKDOWN,
        citation_style=citation_style,
        language=language,
    )
    exporter = PandocExporter(state, config)
    return exporter._generate_markdown()


# Export
__all__ = [
    "ExportFormat",
    "ExportConfig",
    "PandocExporter",
    "export_review",
    "generate_review_markdown",
]

"""PDF Parser MCP Server - Parse academic PDFs to structured Markdown."""

import os
# Fix OpenMP duplicate library issue on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.parsed_document import (
    Citation,
    Equation,
    ParsedDocument,
    Section,
    Table,
)
from utils.config import Config

# Cache directory for parsed PDFs
CACHE_DIR = Config.CACHE_DIR / "parsed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(pdf_path: str) -> Path:
    """Generate cache file path based on PDF content hash."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Hash based on file content for cache key
    with open(pdf_file, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:16]

    return CACHE_DIR / f"{pdf_file.stem}_{file_hash}.json"


def _load_from_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    """Load parsed document from cache if exists."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_to_cache(cache_path: Path, data: Dict[str, Any]) -> None:
    """Save parsed document to cache."""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _extract_sections(markdown: str) -> List[Section]:
    """Extract sections from markdown based on headings."""
    sections = []
    lines = markdown.split("\n")

    current_section = None
    current_content = []
    current_level = 0

    for line in lines:
        # Match markdown headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if heading_match:
            # Save previous section
            if current_section:
                current_section.content = "\n".join(current_content).strip()
                sections.append(current_section)

            # Start new section
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            current_section = Section(
                title=title,
                content="",
                level=level,
                start_page=0,  # Will be updated if page markers exist
                end_page=0,
            )
            current_level = level
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_section:
        current_section.content = "\n".join(current_content).strip()
        sections.append(current_section)

    return sections


def _extract_tables(markdown: str) -> List[Table]:
    """Extract tables from markdown."""
    tables = []
    lines = markdown.split("\n")

    in_table = False
    table_lines = []
    table_caption = ""
    table_count = 0

    for i, line in enumerate(lines):
        # Check for table caption (usually before table)
        if "Table" in line and not in_table:
            # Look for patterns like "Table 1:" or "**Table 1.**"
            caption_match = re.search(r"(Table\s*\d*[.:].+?)(?:\n|$)", line)
            if caption_match:
                table_caption = caption_match.group(1).strip()

        # Detect table start (markdown table with |)
        if "|" in line and not in_table:
            # Check if it's a valid table row
            if line.strip().startswith("|") or re.match(r".*\|.*\|", line):
                in_table = True
                table_lines = [line]
        elif in_table:
            if "|" in line:
                table_lines.append(line)
            else:
                # Table ended
                if len(table_lines) >= 2:  # At least header + separator
                    table_count += 1
                    tables.append(
                        Table(
                            caption=table_caption or f"Table {table_count}",
                            content="\n".join(table_lines),
                            page_num=0,
                            table_id=f"table_{table_count}",
                        )
                    )
                in_table = False
                table_lines = []
                table_caption = ""

    # Handle last table if document ends with table
    if in_table and len(table_lines) >= 2:
        table_count += 1
        tables.append(
            Table(
                caption=table_caption or f"Table {table_count}",
                content="\n".join(table_lines),
                page_num=0,
                table_id=f"table_{table_count}",
            )
        )

    return tables


def _extract_equations(markdown: str) -> List[Equation]:
    """Extract LaTeX equations from markdown."""
    equations = []

    # Match display equations ($$...$$)
    display_pattern = r"\$\$(.+?)\$\$"
    for match in re.finditer(display_pattern, markdown, re.DOTALL):
        latex = match.group(1).strip()
        # Get context (text around equation)
        start = max(0, match.start() - 100)
        end = min(len(markdown), match.end() + 100)
        context = markdown[start:end]

        equations.append(
            Equation(
                latex=latex,
                page_num=0,
                context=context,
                equation_id=f"eq_{len(equations) + 1}",
            )
        )

    # Match inline equations ($...$) - be careful not to match currency
    inline_pattern = r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)"
    for match in re.finditer(inline_pattern, markdown):
        latex = match.group(1).strip()
        # Skip if looks like currency
        if re.match(r"^\d+([,\.]\d+)?$", latex):
            continue

        start = max(0, match.start() - 50)
        end = min(len(markdown), match.end() + 50)
        context = markdown[start:end]

        equations.append(
            Equation(
                latex=latex,
                page_num=0,
                context=context,
                equation_id=f"eq_{len(equations) + 1}",
            )
        )

    return equations


def _extract_citations(markdown: str) -> tuple[List[Citation], List[str]]:
    """Extract in-text citations and references."""
    citations = []
    references = []

    # Extract in-text citations like [1], [2,3], [Smith2024]
    citation_pattern = r"\[([^\]]+)\]"
    for match in re.finditer(citation_pattern, markdown):
        ref_text = match.group(1)
        # Filter out likely non-citations (URLs, figure references)
        if any(x in ref_text.lower() for x in ["http", "fig", "table", "eq"]):
            continue
        # Check if it looks like a citation
        if re.match(r"^\d+$", ref_text) or re.match(r"^[\d,\s-]+$", ref_text):
            start = max(0, match.start() - 100)
            end = min(len(markdown), match.end() + 50)
            context = markdown[start:end]

            citations.append(
                Citation(
                    text=f"[{ref_text}]",
                    page_num=0,
                    context=context,
                    ref_id=ref_text,
                )
            )

    # Extract references section
    ref_section_patterns = [
        r"(?:^|\n)#+\s*References?\s*\n([\s\S]+?)(?=\n#+|\Z)",
        r"(?:^|\n)References?\s*\n[-=]+\s*\n([\s\S]+?)(?=\n#+|\Z)",
    ]

    for pattern in ref_section_patterns:
        ref_match = re.search(pattern, markdown, re.IGNORECASE)
        if ref_match:
            ref_text = ref_match.group(1)
            # Split by numbered entries or newlines
            ref_entries = re.split(r"\n(?=\[\d+\]|\d+\.)", ref_text)
            references = [r.strip() for r in ref_entries if r.strip()]
            break

    return citations, references


def _parse_pdf_with_pymupdf(pdf_path: str) -> str:
    """Parse PDF using pymupdf4llm (fast, stable, no OCR)."""
    try:
        import pymupdf4llm
    except ImportError:
        raise ImportError(
            "pymupdf4llm not installed. Run: pip install pymupdf4llm"
        )

    # Convert PDF to markdown
    markdown = pymupdf4llm.to_markdown(pdf_path)
    return markdown


def _parse_pdf_with_marker(pdf_path: str) -> str:
    """Parse PDF using marker-pdf (slower, has OCR support).

    NOTE: marker-pdf has known issues on macOS MPS. Use with caution.
    Consider using pymupdf4llm as the default backend.
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser
    except ImportError:
        raise ImportError(
            "marker-pdf not installed. Run: pip install marker-pdf"
        )

    # Create model dictionary
    config_parser = ConfigParser({})
    model_dict = create_model_dict()

    # Create converter and parse
    converter = PdfConverter(artifact_dict=model_dict, config=config_parser.generate_config_dict())
    rendered = converter(pdf_path)

    # rendered is a RenderedDocument with markdown property
    return rendered.markdown


def _parse_pdf(pdf_path: str, backend: str = "pymupdf") -> str:
    """Parse PDF with specified backend.

    Args:
        pdf_path: Path to the PDF file
        backend: "pymupdf" (default, fast) or "marker" (OCR support)

    Returns:
        Markdown string
    """
    if backend == "marker":
        return _parse_pdf_with_marker(pdf_path)
    else:
        return _parse_pdf_with_pymupdf(pdf_path)


def _get_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """Get basic PDF metadata using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {"num_pages": 0, "error": "PyMuPDF not installed"}

    try:
        doc = fitz.open(pdf_path)
        metadata = {
            "num_pages": doc.page_count,
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
        }
        doc.close()
        return metadata
    except Exception as e:
        return {"num_pages": 0, "error": str(e)}



async def parse_pdf(
    pdf_path: str,
    use_cache: bool = True,
    extract_tables: bool = True,
    extract_equations: bool = True,
    extract_citations: bool = True,
    backend: str = "pymupdf",
) -> dict:
    """
    Parse an academic PDF into structured Markdown with extracted elements.

    Args:
        pdf_path: Path to the PDF file
        use_cache: Whether to use cached results if available (default: True)
        extract_tables: Extract tables from the document (default: True)
        extract_equations: Extract LaTeX equations (default: True)
        extract_citations: Extract citations and references (default: True)
        backend: Parser backend - "pymupdf" (default, fast) or "marker" (OCR support)

    Returns:
        Dictionary with:
        - markdown: Full document in Markdown format
        - sections: List of document sections with titles and content
        - tables: Extracted tables (if enabled)
        - equations: Extracted equations (if enabled)
        - citations: In-text citations (if enabled)
        - references: Bibliography entries (if enabled)
        - metadata: Document metadata (pages, word count, etc.)
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        return {"error": f"PDF file not found: {pdf_path}"}

    # Check cache
    cache_path = _get_cache_path(pdf_path)
    if use_cache:
        cached = _load_from_cache(cache_path)
        if cached:
            return cached

    loop = asyncio.get_event_loop()

    # Parse PDF (CPU-intensive, run in executor)
    try:
        markdown = await loop.run_in_executor(
            None, _parse_pdf, pdf_path, backend
        )
    except Exception as e:
        return {"error": f"Failed to parse PDF: {str(e)}"}

    # Get PDF metadata
    pdf_metadata = await loop.run_in_executor(
        None, _get_pdf_metadata, pdf_path
    )

    # Extract structured elements
    sections = _extract_sections(markdown)

    tables = []
    if extract_tables:
        tables = _extract_tables(markdown)

    equations = []
    if extract_equations:
        equations = _extract_equations(markdown)

    citations = []
    references = []
    if extract_citations:
        citations, references = _extract_citations(markdown)

    # Build ParsedDocument
    doc = ParsedDocument(
        markdown=markdown,
        sections=sections,
        tables=tables,
        equations=equations,
        citations=citations,
        references=references,
        metadata={
            **pdf_metadata,
            "word_count": len(markdown.split()),
            "source_file": str(pdf_file.absolute()),
        },
    )

    result = doc.to_dict()

    # Cache result
    _save_to_cache(cache_path, result)

    return result



async def extract_section(
    pdf_path: str,
    section_name: str,
    fuzzy_match: bool = True,
) -> dict:
    """
    Extract a specific section from a PDF by name.

    Args:
        pdf_path: Path to the PDF file
        section_name: Name of the section to extract (e.g., "Abstract", "Methods")
        fuzzy_match: Allow partial matching of section names (default: True)

    Returns:
        Dictionary with section title, content, and level
    """
    # First parse the full document
    result = await parse_pdf(pdf_path, extract_tables=False, extract_equations=False)

    if "error" in result:
        return result

    sections = result.get("sections", [])
    section_lower = section_name.lower()

    for section in sections:
        title_lower = section["title"].lower()
        if fuzzy_match:
            if section_lower in title_lower or title_lower in section_lower:
                return {
                    "found": True,
                    "title": section["title"],
                    "content": section["content"],
                    "level": section["level"],
                }
        else:
            if title_lower == section_lower:
                return {
                    "found": True,
                    "title": section["title"],
                    "content": section["content"],
                    "level": section["level"],
                }

    return {
        "found": False,
        "error": f"Section '{section_name}' not found",
        "available_sections": [s["title"] for s in sections],
    }



async def extract_all_tables(pdf_path: str) -> dict:
    """
    Extract all tables from a PDF document.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with list of tables (caption, content in Markdown, page)
    """
    result = await parse_pdf(
        pdf_path,
        extract_tables=True,
        extract_equations=False,
        extract_citations=False,
    )

    if "error" in result:
        return result

    tables = result.get("tables", [])

    return {
        "count": len(tables),
        "tables": tables,
    }



async def extract_all_citations(pdf_path: str) -> dict:
    """
    Extract all citations and references from a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with in-text citations and bibliography references
    """
    result = await parse_pdf(
        pdf_path,
        extract_tables=False,
        extract_equations=False,
        extract_citations=True,
    )

    if "error" in result:
        return result

    return {
        "citation_count": len(result.get("citations", [])),
        "citations": result.get("citations", []),
        "reference_count": len(result.get("references", [])),
        "references": result.get("references", []),
    }



async def get_document_info(pdf_path: str) -> dict:
    """
    Get quick document information without full parsing.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with page count, section list, and basic metadata
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        return {"error": f"PDF file not found: {pdf_path}"}

    loop = asyncio.get_event_loop()

    # Get PDF metadata (fast)
    pdf_metadata = await loop.run_in_executor(
        None, _get_pdf_metadata, pdf_path
    )

    # Check if we have cached full parse
    cache_path = _get_cache_path(pdf_path)
    cached = _load_from_cache(cache_path)

    if cached:
        sections = [s["title"] for s in cached.get("sections", [])]
        return {
            "file_name": pdf_file.name,
            "file_size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2),
            "num_pages": pdf_metadata.get("num_pages", 0),
            "title": pdf_metadata.get("title", ""),
            "author": pdf_metadata.get("author", ""),
            "sections": sections,
            "table_count": len(cached.get("tables", [])),
            "equation_count": len(cached.get("equations", [])),
            "citation_count": len(cached.get("citations", [])),
            "word_count": cached.get("metadata", {}).get("word_count", 0),
            "cached": True,
        }

    return {
        "file_name": pdf_file.name,
        "file_size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2),
        "num_pages": pdf_metadata.get("num_pages", 0),
        "title": pdf_metadata.get("title", ""),
        "author": pdf_metadata.get("author", ""),
        "cached": False,
        "note": "Full parse not cached. Use parse_pdf for complete analysis.",
    }



async def clear_cache(pdf_path: Optional[str] = None) -> dict:
    """
    Clear parsed PDF cache.

    Args:
        pdf_path: Clear cache for specific PDF. If None, clears all cache.

    Returns:
        Dictionary with number of cache files removed
    """
    if pdf_path:
        try:
            cache_path = _get_cache_path(pdf_path)
            if cache_path.exists():
                cache_path.unlink()
                return {"cleared": 1, "file": str(cache_path)}
            return {"cleared": 0, "message": "No cache found for this file"}
        except FileNotFoundError:
            return {"cleared": 0, "error": "PDF file not found"}
    else:
        # Clear all cache
        count = 0
        for cache_file in CACHE_DIR.glob("*.json"):
            cache_file.unlink()
            count += 1
        return {"cleared": count, "message": f"Cleared {count} cached files"}



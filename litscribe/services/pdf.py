"""PDF download + parse service — ported from src/services/pdf_parser.py.

Uses pymupdf4llm to convert academic PDFs into structured Markdown.
Downloads from URL, caches parsed results by content hash.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

from litscribe.models.analysis import ParsedDoc

logger = logging.getLogger(__name__)


class PDFService:
    """PDF download and parsing service.

    Args:
        cache_dir: Directory for caching parsed results.
                   Defaults to a ``pdf_parsed`` subdirectory next to the
                   caller's data directory; pass explicitly from Config.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path(tempfile.gettempdir()) / "litscribe_pdf_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(self, url: str, paper_id: str) -> ParsedDoc:
        """Download a PDF from *url* and return a :class:`ParsedDoc`.

        The raw PDF is downloaded to a temp file, parsed via pymupdf4llm,
        and cached by content hash so repeated calls are fast.
        """
        pdf_bytes = await self._download(url)
        content_hash = hashlib.md5(pdf_bytes).hexdigest()[:16]

        # Check cache
        cache_path = self._cache_dir / f"{content_hash}.json"
        cached = self._load_cache(cache_path)
        if cached is not None:
            logger.debug("PDF cache hit for %s", paper_id)
            return ParsedDoc(
                paper_id=paper_id,
                markdown=cached["markdown"],
                sections=cached["sections"],
                word_count=cached["word_count"],
            )

        # Write to temp file for pymupdf4llm (needs a path)
        tmp = Path(tempfile.mktemp(suffix=".pdf"))
        try:
            tmp.write_bytes(pdf_bytes)
            markdown = await self._parse_with_pymupdf(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)

        sections = _extract_sections(markdown)
        word_count = len(markdown.split())

        # Cache result
        self._save_cache(cache_path, {
            "markdown": markdown,
            "sections": sections,
            "word_count": word_count,
        })

        return ParsedDoc(
            paper_id=paper_id,
            markdown=markdown,
            sections=sections,
            word_count=word_count,
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    @staticmethod
    async def _download(url: str) -> bytes:
        """Download PDF bytes from *url*."""
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    async def _parse_with_pymupdf(pdf_path: str) -> str:
        """Parse a local PDF file to Markdown using pymupdf4llm (CPU-bound)."""
        import pymupdf4llm  # type: ignore[import-untyped]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, pymupdf4llm.to_markdown, pdf_path)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_cache(path: Path) -> dict[str, Any] | None:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return None

    @staticmethod
    def _save_cache(path: Path, data: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)


# ------------------------------------------------------------------
# Markdown section extraction (ported from src/services/pdf_parser.py)
# ------------------------------------------------------------------

def _extract_sections(markdown: str) -> list[dict]:
    """Split Markdown into a list of ``{title, content, level}`` dicts."""
    sections: list[dict] = []
    current_title: str | None = None
    current_level = 0
    current_lines: list[str] = []

    for line in markdown.split("\n"):
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            if current_title is not None:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines).strip(),
                    "level": current_level,
                })
            current_level = len(heading.group(1))
            current_title = heading.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_title is not None:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines).strip(),
            "level": current_level,
        })

    return sections

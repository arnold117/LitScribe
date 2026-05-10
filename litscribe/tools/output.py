from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "~/.litscribe/output"


def get_output_dir() -> Path:
    custom = os.getenv("LITSCRIBE_OUTPUT_DIR", "")
    if custom:
        d = Path(custom).expanduser().resolve()
    else:
        d = Path(_DEFAULT_DIR).expanduser().resolve()

    try:
        d.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback = Path.home() / ".litscribe" / "output"
        logger.warning(f"No write permission to {d}, falling back to {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    # Verify writable
    test_file = d / ".write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
    except (PermissionError, OSError):
        fallback = Path.home() / ".litscribe" / "output"
        logger.warning(f"Directory {d} is not writable, falling back to {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    return d


def _safe_filename(question: str) -> str:
    name = question[:40].replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_一-鿿-]", "", name)
    return name or "review"


def save_review(text: str, question: str, confirm: bool = True) -> str:
    output_dir = get_output_dir()
    filename = f"{_safe_filename(question)}.md"
    filepath = output_dir / filename

    # Check if file exists — version it
    if filepath.exists():
        i = 2
        while filepath.exists():
            filepath = output_dir / f"{_safe_filename(question)}_v{i}.md"
            i += 1

    filepath.write_text(text, encoding="utf-8")
    logger.info(f"Review saved to {filepath}")
    return str(filepath)


def list_output_files() -> list[dict]:
    output_dir = get_output_dir()
    files = []
    for f in sorted(output_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        files.append({
            "path": str(f),
            "name": f.name,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime,
        })
    return files

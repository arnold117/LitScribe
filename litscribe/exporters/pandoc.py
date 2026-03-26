"""Pandoc exporter. Port from src/exporters/pandoc_exporter.py."""
from __future__ import annotations


def export_pandoc(text: str, output_path: str, format: str = "docx") -> str:
    """Export review text via Pandoc to the given format.

    Port from src/exporters/pandoc_exporter.py.
    """
    raise NotImplementedError("Port from src/exporters/pandoc_exporter.py")

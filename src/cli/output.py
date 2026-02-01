"""Unified output manager for console and log file output.

This module provides a consistent way to output messages to both the console
and log files, ensuring all user-facing output is captured in logs for
debugging and reproducibility.
"""

import logging
from typing import Any, Dict, List, Optional


class OutputManager:
    """Unified output manager for dual console/log output.

    All user-facing output should go through this class to ensure
    it's captured in both the console and log file.

    Usage:
        from cli.output import get_output
        out = get_output(__name__)
        out.header("Starting Process")
        out.info("Processing 10 papers...")
        out.success("Complete!")
    """

    def __init__(self, logger: logging.Logger):
        """Initialize OutputManager with a logger.

        Args:
            logger: Logger instance for file output
        """
        self.logger = logger

    def info(self, msg: str) -> None:
        """Output informational message.

        Args:
            msg: Message to output
        """
        print(msg)
        self.logger.info(msg)

    def success(self, msg: str, emoji: str = "✓") -> None:
        """Output success message with optional emoji.

        Args:
            msg: Message to output
            emoji: Emoji prefix (default: ✓)
        """
        print(f"{emoji} {msg}")
        self.logger.info(f"[SUCCESS] {msg}")

    def warning(self, msg: str, emoji: str = "⚠") -> None:
        """Output warning message.

        Args:
            msg: Message to output
            emoji: Emoji prefix (default: ⚠)
        """
        print(f"{emoji} {msg}")
        self.logger.warning(msg)

    def error(self, msg: str, emoji: str = "❌") -> None:
        """Output error message.

        Args:
            msg: Message to output
            emoji: Emoji prefix (default: ❌)
        """
        print(f"{emoji} {msg}")
        self.logger.error(msg)

    def header(self, title: str, width: int = 60) -> None:
        """Output section header.

        Args:
            title: Header title
            width: Width of the separator line
        """
        line = "=" * width
        print(f"\n{line}")
        print(f"  {title}")
        print(f"{line}\n")
        self.logger.info(f"=== {title} ===")

    def subheader(self, title: str, emoji: str = "") -> None:
        """Output subsection header.

        Args:
            title: Subsection title
            emoji: Optional emoji prefix
        """
        prefix = f"{emoji} " if emoji else ""
        print(f"\n{prefix}{title}:")
        self.logger.info(f"--- {title} ---")

    def stat(self, label: str, value: Any, indent: int = 3) -> None:
        """Output a statistic or key-value pair.

        Args:
            label: Statistic label
            value: Statistic value
            indent: Number of spaces to indent
        """
        spaces = " " * indent
        print(f"{spaces}{label}: {value}")
        self.logger.info(f"STAT {label}={value}")

    def stats(self, stats_dict: Dict[str, Any], indent: int = 3) -> None:
        """Output multiple statistics.

        Args:
            stats_dict: Dictionary of label -> value pairs
            indent: Number of spaces to indent
        """
        for label, value in stats_dict.items():
            self.stat(label, value, indent)

    def bullet(self, msg: str, indent: int = 3) -> None:
        """Output a bullet point item.

        Args:
            msg: Bullet point text
            indent: Number of spaces to indent
        """
        spaces = " " * indent
        print(f"{spaces}- {msg}")
        self.logger.info(f"  - {msg}")

    def numbered(self, items: List[str], start: int = 1, indent: int = 3) -> None:
        """Output a numbered list.

        Args:
            items: List of items
            start: Starting number
            indent: Number of spaces to indent
        """
        spaces = " " * indent
        for i, item in enumerate(items, start):
            print(f"{spaces}{i}. {item}")
            self.logger.info(f"  {i}. {item}")

    def preview(
        self,
        title: str,
        content: str,
        max_chars: int = 500,
        width: int = 60,
    ) -> None:
        """Output a content preview with truncation.

        Args:
            title: Preview title
            content: Content to preview
            max_chars: Maximum characters to show
            width: Width of the separator line
        """
        line = "=" * width
        print(f"\n{line}")
        print(title)
        print(line)
        print(content[:max_chars])
        if len(content) > max_chars:
            print(f"\n... [{len(content) - max_chars} more chars]")
        self.logger.info(f"PREVIEW: {title} ({len(content)} chars total)")

    def table_row(
        self,
        columns: List[str],
        widths: Optional[List[int]] = None,
        indent: int = 0,
    ) -> None:
        """Output a formatted table row.

        Args:
            columns: List of column values
            widths: Optional list of column widths (for padding)
            indent: Number of spaces to indent
        """
        spaces = " " * indent
        if widths:
            formatted = []
            for col, width in zip(columns, widths):
                formatted.append(str(col).ljust(width))
            row = " | ".join(formatted)
        else:
            row = " | ".join(str(col) for col in columns)

        print(f"{spaces}{row}")
        self.logger.info(f"TABLE: {row}")

    def progress(self, current: int, total: int, task: str = "") -> None:
        """Output progress indicator.

        Args:
            current: Current item number
            total: Total items
            task: Optional task description
        """
        pct = (current / total * 100) if total > 0 else 0
        task_str = f" {task}" if task else ""
        msg = f"[{current}/{total}] ({pct:.0f}%){task_str}"
        print(f"\r{msg}", end="", flush=True)
        # Only log at 25% intervals or completion
        if current == total or current % max(1, total // 4) == 0:
            self.logger.info(f"PROGRESS: {msg}")

    def progress_done(self) -> None:
        """Complete progress line with newline."""
        print()  # Just add newline after progress

    def blank(self) -> None:
        """Output a blank line."""
        print()

    def divider(self, char: str = "-", width: int = 60) -> None:
        """Output a divider line.

        Args:
            char: Character to use for the divider
            width: Width of the divider
        """
        print(char * width)
        self.logger.info(f"--- divider ---")

    def paper(
        self,
        paper: Dict[str, Any],
        index: int,
        verbose: bool = False,
    ) -> None:
        """Output a paper summary.

        Args:
            paper: Paper dictionary
            index: Paper index number
            verbose: Whether to show detailed info
        """
        title = paper.get("title", "Unknown")
        if len(title) > 70:
            title = title[:67] + "..."

        print(f"{index}. {title}")
        self.logger.info(f"PAPER {index}: {title}")

        authors = paper.get("authors", [])
        if authors:
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " ..."
            print(f"   Authors: {author_str}")
            self.logger.info(f"   Authors: {author_str}")

        year = paper.get("year", "N/A")
        citations = paper.get("citations", 0)
        sources = list(paper.get("sources", {}).keys()) if paper.get("sources") else []
        info_line = f"   Year: {year} | Citations: {citations}"
        if sources:
            info_line += f" | Sources: {sources}"
        print(info_line)
        self.logger.info(info_line)

        if verbose:
            if paper.get("doi"):
                print(f"   DOI: {paper['doi']}")
                self.logger.info(f"   DOI: {paper['doi']}")
            if paper.get("arxiv_id"):
                print(f"   arXiv: {paper['arxiv_id']}")
                self.logger.info(f"   arXiv: {paper['arxiv_id']}")
            if paper.get("abstract"):
                abstract = paper["abstract"]
                if len(abstract) > 200:
                    abstract = abstract[:200] + "..."
                print(f"   Abstract: {abstract}")
                self.logger.info(f"   Abstract: {abstract[:100]}...")

        print()


# Global output manager registry
_output_managers: Dict[str, OutputManager] = {}


def get_output(name: str = "litscribe") -> OutputManager:
    """Get or create an OutputManager for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        OutputManager instance
    """
    if name not in _output_managers:
        logger = logging.getLogger(name)
        _output_managers[name] = OutputManager(logger)
    return _output_managers[name]


def create_output(logger: logging.Logger) -> OutputManager:
    """Create an OutputManager with a specific logger.

    Args:
        logger: Logger instance

    Returns:
        OutputManager instance
    """
    return OutputManager(logger)


__all__ = ["OutputManager", "get_output", "create_output"]

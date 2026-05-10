from __future__ import annotations

import difflib
from typing import Any


def unified_diff(old: str, new: str, old_name: str = "before", new_name: str = "after") -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff = difflib.unified_diff(old_lines, new_lines, fromfile=old_name, tofile=new_name)
    return "".join(diff)


def colored_diff(old: str, new: str) -> str:
    old_lines = old.splitlines()
    new_lines = new.splitlines()

    diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
    lines = []
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            lines.append(f"\033[1m{line}\033[0m")
        elif line.startswith("+"):
            lines.append(f"\033[32m{line}\033[0m")
        elif line.startswith("-"):
            lines.append(f"\033[31m{line}\033[0m")
        elif line.startswith("@@"):
            lines.append(f"\033[36m{line}\033[0m")
        else:
            lines.append(line)

    return "\n".join(lines)


def diff_stats(old: str, new: str) -> dict[str, int]:
    old_lines = old.splitlines()
    new_lines = new.splitlines()

    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))

    return {"added": added, "removed": removed, "unchanged": len(new_lines) - added}


def html_diff(old: str, new: str) -> str:
    old_lines = old.splitlines()
    new_lines = new.splitlines()

    differ = difflib.HtmlDiff(wrapcolumn=80)
    return differ.make_table(old_lines, new_lines, fromdesc="Before", todesc="After")

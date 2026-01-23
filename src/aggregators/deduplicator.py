"""Paper deduplication logic for unified search."""

import re
from difflib import SequenceMatcher
from typing import List, Tuple

from models.unified_paper import UnifiedPaper


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    # Lowercase
    title = title.lower()
    # Remove common prefixes/suffixes
    title = re.sub(r"^(a|an|the)\s+", "", title)
    # Remove punctuation and extra whitespace
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles (0-1)."""
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)

    # Exact match after normalization
    if norm1 == norm2:
        return 1.0

    # Use SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, norm1, norm2).ratio()


def are_same_paper(paper1: UnifiedPaper, paper2: UnifiedPaper) -> bool:
    """
    Determine if two papers are the same based on identifiers and title.

    Returns True if papers should be merged.
    """
    # 1. Exact identifier match (highest confidence)
    if paper1.doi and paper2.doi:
        if paper1.doi.lower() == paper2.doi.lower():
            return True

    if paper1.arxiv_id and paper2.arxiv_id:
        # Normalize arxiv IDs (remove version suffix for comparison)
        id1 = paper1.arxiv_id.split("v")[0]
        id2 = paper2.arxiv_id.split("v")[0]
        if id1 == id2:
            return True

    if paper1.pmid and paper2.pmid:
        if paper1.pmid == paper2.pmid:
            return True

    # 2. Title similarity (requires high threshold)
    sim = title_similarity(paper1.title, paper2.title)
    if sim >= 0.92:
        # Additional check: year should match (if both have year)
        if paper1.year and paper2.year:
            if paper1.year == paper2.year:
                return True
        else:
            # No year info, rely on title similarity alone
            return True

    # 3. Very high title similarity with overlapping authors
    if sim >= 0.85:
        # Check if any author overlaps
        authors1 = {a.lower() for a in paper1.authors}
        authors2 = {a.lower() for a in paper2.authors}

        # Check for last name matches
        last_names1 = {a.split()[-1].lower() if a.split() else a.lower() for a in paper1.authors}
        last_names2 = {a.split()[-1].lower() if a.split() else a.lower() for a in paper2.authors}

        if last_names1 & last_names2:  # Intersection not empty
            return True

    return False


def deduplicate_papers(papers: List[UnifiedPaper]) -> List[UnifiedPaper]:
    """
    Deduplicate a list of papers, merging duplicate entries.

    Args:
        papers: List of UnifiedPaper objects

    Returns:
        Deduplicated list with merged information
    """
    if not papers:
        return []

    # Group papers that are the same
    groups: List[List[UnifiedPaper]] = []
    used = set()

    for i, paper in enumerate(papers):
        if i in used:
            continue

        group = [paper]
        used.add(i)

        for j, other in enumerate(papers[i + 1 :], start=i + 1):
            if j in used:
                continue
            if are_same_paper(paper, other):
                group.append(other)
                used.add(j)

        groups.append(group)

    # Merge each group into a single paper
    result = []
    for group in groups:
        if len(group) == 1:
            merged = group[0]
        else:
            # Merge all papers in the group
            merged = group[0]
            for other in group[1:]:
                merged = merged.merge_with(other)

        # Recalculate completeness score
        merged.calculate_completeness_score()
        result.append(merged)

    return result


def rank_papers(
    papers: List[UnifiedPaper],
    sort_by: str = "relevance",
) -> List[UnifiedPaper]:
    """
    Rank papers by specified criteria.

    Args:
        papers: List of papers to rank
        sort_by: Ranking criteria - "relevance", "citations", "year", "completeness"

    Returns:
        Sorted list of papers
    """
    if sort_by == "citations":
        return sorted(papers, key=lambda p: p.citations or 0, reverse=True)
    elif sort_by == "year":
        return sorted(papers, key=lambda p: p.year or 0, reverse=True)
    elif sort_by == "completeness":
        return sorted(papers, key=lambda p: p.completeness_score, reverse=True)
    else:  # relevance (default)
        # Combine relevance and completeness scores
        return sorted(
            papers,
            key=lambda p: (p.relevance_score * 0.7 + p.completeness_score * 0.3),
            reverse=True,
        )

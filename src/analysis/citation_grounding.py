"""Citation grounding check for literature reviews.

Verifies that inline citations [Author, Year] in the generated review text
actually correspond to papers that were analyzed in the pipeline.
Ungrounded citations are flagged as potential hallucinations.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# Regex patterns for inline citation formats
# Matches: [Smith, 2020], [Smith et al., 2020], [Smith & Jones, 2020]
# Also: [Smith, 2020; Jones, 2021] (multi-citation)
CITATION_PATTERN = re.compile(
    r'\[([A-Z][a-zà-ÿ]+(?:\s(?:&|and)\s[A-Z][a-zà-ÿ]+)?'
    r'(?:\set\sal\.)?,?\s*\d{4}'
    r'(?:;\s*[A-Z][a-zà-ÿ]+(?:\s(?:&|and)\s[A-Z][a-zà-ÿ]+)?'
    r'(?:\set\sal\.)?,?\s*\d{4})*)\]'
)

# Split multi-citations: "Smith, 2020; Jones, 2021" -> individual citations
MULTI_SPLIT = re.compile(r';\s*')

# Parse a single citation into (author_part, year)
SINGLE_CITATION = re.compile(
    r'([A-Z][a-zà-ÿ]+(?:\s(?:&|and)\s[A-Z][a-zà-ÿ]+)?(?:\set\sal\.)?),?\s*(\d{4})'
)


def extract_inline_citations(review_text: str) -> List[str]:
    """Extract all inline citations from review text.

    Args:
        review_text: The generated literature review text

    Returns:
        List of citation strings, e.g. ["Smith, 2020", "Jones et al., 2021"]
    """
    citations = []
    for match in CITATION_PATTERN.finditer(review_text):
        group = match.group(1)
        # Split multi-citations
        parts = MULTI_SPLIT.split(group)
        for part in parts:
            part = part.strip()
            if part and SINGLE_CITATION.match(part):
                citations.append(part)
    return citations


def _extract_last_names(authors: List[str]) -> List[str]:
    """Extract last names from author strings.

    Handles formats like:
    - "John Smith" -> "Smith"
    - "Smith, J." -> "Smith"
    - "J. Smith" -> "Smith"
    """
    last_names = []
    for author in authors:
        author = author.strip()
        if not author:
            continue
        if "," in author:
            # "Smith, J." format
            last_names.append(author.split(",")[0].strip())
        else:
            # "John Smith" or "J. Smith" format
            parts = author.split()
            if parts:
                last_names.append(parts[-1].strip())
    return last_names


def _parse_citation(citation: str) -> Optional[Tuple[str, str]]:
    """Parse a citation string into (author_part, year).

    Returns:
        Tuple of (author_last_name_or_phrase, year_str) or None
    """
    m = SINGLE_CITATION.match(citation.strip())
    if m:
        return (m.group(1).strip(), m.group(2).strip())
    return None


def _normalize(s: str) -> str:
    """Normalize a string for fuzzy comparison."""
    return s.lower().replace(".", "").replace(",", "").strip()


def match_citations_to_papers(
    citations: List[str],
    analyzed_papers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Match each citation to an analyzed paper.

    Args:
        citations: List of citation strings from the review
        analyzed_papers: List of paper dicts with authors, year, title, paper_id

    Returns:
        Dict with grounded, ungrounded lists, and grounding_rate
    """
    grounded = []
    ungrounded = []
    seen_citations = set()

    # Build lookup: (normalized_last_name, year_str) -> paper
    paper_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for paper in analyzed_papers:
        year = str(paper.get("year", ""))
        authors = paper.get("authors", [])
        if isinstance(authors, str):
            authors = [authors]
        last_names = _extract_last_names(authors)
        for ln in last_names:
            key = (_normalize(ln), year)
            paper_lookup[key] = paper

    for citation in citations:
        # Deduplicate
        if citation in seen_citations:
            continue
        seen_citations.add(citation)

        parsed = _parse_citation(citation)
        if not parsed:
            ungrounded.append(citation)
            continue

        author_part, year = parsed
        # Extract the primary last name from citation
        # "Smith et al." -> "Smith", "Smith & Jones" -> "Smith"
        primary_name = author_part.split()[0].replace(",", "")

        key = (_normalize(primary_name), year)
        if key in paper_lookup:
            paper = paper_lookup[key]
            grounded.append({
                "citation": citation,
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
            })
        else:
            # Fuzzy: try all papers with matching year
            matched = False
            for paper in analyzed_papers:
                if str(paper.get("year", "")) != year:
                    continue
                authors = paper.get("authors", [])
                if isinstance(authors, str):
                    authors = [authors]
                last_names = _extract_last_names(authors)
                for ln in last_names:
                    if _normalize(ln) == _normalize(primary_name):
                        grounded.append({
                            "citation": citation,
                            "paper_id": paper.get("paper_id", ""),
                            "title": paper.get("title", ""),
                        })
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                ungrounded.append(citation)

    total = len(seen_citations)
    grounding_rate = len(grounded) / total if total > 0 else 1.0

    return {
        "grounded": grounded,
        "ungrounded": ungrounded,
        "total_citations": total,
        "grounded_count": len(grounded),
        "ungrounded_count": len(ungrounded),
        "grounding_rate": round(grounding_rate, 4),
    }


def check_citation_grounding(
    review_text: str,
    analyzed_papers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Full citation grounding check pipeline.

    Args:
        review_text: Generated review text
        analyzed_papers: List of paper dicts from the pipeline

    Returns:
        Complete grounding report
    """
    citations = extract_inline_citations(review_text)
    return match_citations_to_papers(citations, analyzed_papers)

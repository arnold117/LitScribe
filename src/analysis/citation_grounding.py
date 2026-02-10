"""Citation grounding check for literature reviews.

Verifies that inline citations [Author, Year] in the generated review text
actually correspond to papers that were analyzed in the pipeline.
Ungrounded citations are flagged as potential hallucinations.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# Regex patterns for inline citation formats
# Matches: [Smith, 2020], [Smith et al., 2020], [Smith & Jones, 2020]
# Also Chinese: [Ma等, 2006], [Zhang和Li, 2020]
# Also multi: [Smith, 2020; Jones, 2021]
_AUTHOR = r'[A-Z][a-zà-ÿ]+'
_ET_AL = r'(?:\set\sal\.|\s*等)'  # "et al." or Chinese "等"
_AND = r'(?:\s(?:&|and|和)\s)'    # "&", "and", or Chinese "和"
_SINGLE = rf'{_AUTHOR}(?:{_AND}{_AUTHOR})?{_ET_AL}?'
_YEAR = r',?\s*\d{4}'

CITATION_PATTERN = re.compile(
    rf'\[({_SINGLE}{_YEAR}(?:;\s*{_SINGLE}{_YEAR})*)\]'
)

# Year-less citations: [Smith et al.], [Zhang等], [Smith & Jones]
# These are common when LLMs ignore the year requirement
YEARLESS_PATTERN = re.compile(
    rf'\[({_SINGLE})\]'
)

# Split multi-citations: "Smith, 2020; Jones, 2021" -> individual citations
MULTI_SPLIT = re.compile(r';\s*')

# Parse a single citation into (author_part, year)
SINGLE_CITATION = re.compile(
    rf'({_AUTHOR}(?:{_AND}{_AUTHOR})?{_ET_AL}?),?\s*(\d{{4}})'
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


def extract_yearless_citations(review_text: str) -> List[str]:
    """Extract year-less citations like [Author et al.] from review text.

    These are emitted when LLMs ignore the [Author, Year] format requirement.

    Args:
        review_text: The generated literature review text

    Returns:
        List of author-only citation strings, e.g. ["Smith et al.", "Zhang等"]
    """
    # First collect all year-bearing citation spans to exclude them
    year_spans = set()
    for match in CITATION_PATTERN.finditer(review_text):
        year_spans.add((match.start(), match.end()))

    citations = []
    for match in YEARLESS_PATTERN.finditer(review_text):
        # Skip if this span overlaps with a year-bearing citation
        if any(match.start() >= s and match.end() <= e for s, e in year_spans):
            continue
        group = match.group(1).strip()
        if group:
            citations.append(group)
    return citations


def extract_all_cited_authors(review_text: str) -> set:
    """Extract all cited author last names from review text.

    Handles both [Author, Year] and [Author et al.] citation formats.

    Args:
        review_text: The generated literature review text

    Returns:
        Set of lowercase author last names found in citations
    """
    authors = set()

    # From year-bearing citations
    for cit in extract_inline_citations(review_text):
        parsed = _parse_citation(cit)
        if parsed:
            author_part, _ = parsed
            primary = author_part.split()[0].replace(",", "").rstrip("等")
            if primary:
                authors.add(primary.lower())

    # From year-less citations
    for cit in extract_yearless_citations(review_text):
        # Extract primary name: "Smith et al." -> "Smith", "Zhang等" -> "Zhang"
        primary = cit.split()[0].replace(",", "").rstrip("等")
        if primary and primary[0].isupper():
            authors.add(primary.lower())

    return authors


def _extract_last_names(authors: List[str]) -> List[str]:
    """Extract last names from author strings.

    Handles formats like:
    - "John Smith" -> "Smith"
    - "Smith, J." -> "Smith"
    - "J. Smith" -> "Smith"
    - "Ma X" -> "Ma"  (PubMed Chinese: LastName + Initials)
    - "Zhang ZJ" -> "Zhang"
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
            parts = author.split()
            if not parts:
                continue
            last_part = parts[-1].rstrip(".")
            # PubMed "LastName Initials" format: last word is all uppercase
            # and short (1-3 chars), e.g. "Ma X", "Zhang ZJ", "Gang DR"
            if len(parts) > 1 and last_part.isupper() and len(last_part) <= 3:
                last_names.append(parts[0].strip())
            else:
                # "John Smith" or "J. Smith" format
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
        # "Smith et al." -> "Smith", "Smith & Jones" -> "Smith", "Ma等" -> "Ma"
        primary_name = author_part.split()[0].replace(",", "").rstrip("等")

        # Detect generic placeholder names that LLMs sometimes emit
        _GENERIC_NAMES = {"author", "authors", "et"}
        is_generic = _normalize(primary_name) in _GENERIC_NAMES

        key = (_normalize(primary_name), year)
        if not is_generic and key in paper_lookup:
            paper = paper_lookup[key]
            grounded.append({
                "citation": citation,
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
            })
        else:
            # Fuzzy: try all papers with matching year
            matched = False
            year_matches = [p for p in analyzed_papers if str(p.get("year", "")) == year]

            if not is_generic:
                # Try author name match
                for paper in year_matches:
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

            # Fallback for generic names: if exactly one paper matches the year, ground it
            if not matched and is_generic and len(year_matches) == 1:
                paper = year_matches[0]
                grounded.append({
                    "citation": citation,
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                })
                matched = True

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

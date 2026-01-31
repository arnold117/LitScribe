"""Critical Reading Agent for LitScribe.

This agent is responsible for:
1. PDF acquisition - downloading from arXiv or getting from Zotero
2. PDF parsing - extracting structured text from PDFs
3. Finding extraction - identifying key findings using LLM
4. Methodology analysis - summarizing research methods
5. Quality assessment - identifying strengths and limitations

Supports PDF and parse caching with SQLite (Phase 6.5).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.errors import AgentError, ErrorType, LLMError, PDFNotFoundError, PDFParseError
from agents.prompts import (
    KEY_FINDINGS_PROMPT,
    METHODOLOGY_ANALYSIS_PROMPT,
    QUALITY_ASSESSMENT_PROMPT,
)
from agents.state import LitScribeState, PaperSummary
from agents.tools import (
    call_llm,
    download_arxiv_pdf,
    extract_pdf_section,
    get_zotero_pdf_path,
    parse_pdf,
)
from cache.cached_tools import CachedTools, get_cached_tools

logger = logging.getLogger(__name__)


async def acquire_pdf(
    paper: Dict[str, Any],
    cached_tools: Optional[CachedTools] = None,
) -> Optional[str]:
    """Acquire PDF file for a paper from available sources.

    Supports caching: checks local cache first before downloading.

    Tries multiple sources in order:
    1. Local cache (if caching enabled)
    2. arXiv (if arXiv ID available)
    3. Zotero (if zotero_key available)
    4. Direct URL download (future)

    Args:
        paper: Paper metadata dict
        cached_tools: CachedTools instance for caching (optional)

    Returns:
        Local path to PDF file, or None if not available
    """
    paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi", "unknown")

    # Use cached tools if available
    if cached_tools and cached_tools.cache_enabled:
        try:
            pdf_path = await cached_tools.get_pdf_with_cache(paper)
            if pdf_path and pdf_path.exists():
                logger.info(f"Got PDF for {paper_id} (cached)")
                return str(pdf_path)
        except Exception as e:
            logger.warning(f"Cache lookup failed for {paper_id}: {e}")
            # Fall through to regular acquisition

    # Regular acquisition without caching
    arxiv_id = paper.get("arxiv_id")
    zotero_key = paper.get("zotero_key")

    # Try arXiv first
    if arxiv_id:
        try:
            result = await download_arxiv_pdf(arxiv_id)
            pdf_path = result.get("pdf_path")
            if pdf_path:
                logger.info(f"Downloaded PDF from arXiv: {arxiv_id}")
                return pdf_path
        except PDFNotFoundError as e:
            logger.warning(f"arXiv PDF not found for {arxiv_id}: {e}")
        except Exception as e:
            logger.warning(f"Failed to download arXiv PDF {arxiv_id}: {e}")

    # Try Zotero
    if zotero_key:
        try:
            result = await get_zotero_pdf_path(zotero_key)
            pdf_path = result.get("pdf_path")
            if pdf_path:
                logger.info(f"Got PDF from Zotero: {zotero_key}")
                return pdf_path
        except PDFNotFoundError as e:
            logger.warning(f"Zotero PDF not found for {zotero_key}: {e}")
        except Exception as e:
            logger.warning(f"Failed to get Zotero PDF {zotero_key}: {e}")

    logger.info(f"No PDF available for paper {paper_id}")
    return None


async def parse_paper_pdf(
    pdf_path: str,
    paper_id: Optional[str] = None,
    cached_tools: Optional[CachedTools] = None,
) -> Dict[str, Any]:
    """Parse a PDF file into structured content.

    Supports caching: stores parsed results for reuse.

    Args:
        pdf_path: Path to the PDF file
        paper_id: Paper identifier for caching (optional)
        cached_tools: CachedTools instance for caching (optional)

    Returns:
        Parsed document with markdown, sections, tables, references
    """
    # Use cached tools if available and paper_id provided
    if cached_tools and cached_tools.cache_enabled and paper_id:
        try:
            result = await cached_tools.parse_pdf_with_cache(
                paper_id=paper_id,
                pdf_path=Path(pdf_path),
            )
            logger.info(f"Parsed PDF for {paper_id} (with cache)")
            return result
        except Exception as e:
            logger.warning(f"Cached parsing failed for {paper_id}: {e}")
            # Fall through to regular parsing

    # Regular parsing without caching
    try:
        # Try pymupdf first (faster, more reliable)
        result = await parse_pdf(pdf_path, backend="pymupdf")
        logger.info(f"Parsed PDF with pymupdf: {pdf_path}")
        return result
    except PDFParseError:
        # Fall back to marker if pymupdf fails
        try:
            result = await parse_pdf(pdf_path, backend="marker")
            logger.info(f"Parsed PDF with marker: {pdf_path}")
            return result
        except PDFParseError as e:
            logger.error(f"Failed to parse PDF {pdf_path}: {e}")
            raise


async def extract_key_findings(
    paper: Dict[str, Any],
    parsed_doc: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> List[str]:
    """Extract key findings from a paper using LLM.

    Args:
        paper: Paper metadata with abstract
        parsed_doc: Parsed PDF content (optional)
        model: LLM model to use

    Returns:
        List of key findings
    """
    title = paper.get("title", "Unknown")
    authors = paper.get("authors", [])
    if isinstance(authors, list):
        authors = ", ".join(authors[:5])
    year = paper.get("year", "N/A")
    abstract = paper.get("abstract", "")

    # Use parsed content if available, otherwise just abstract
    full_text = ""
    if parsed_doc:
        full_text = parsed_doc.get("markdown", "")[:8000]  # Limit to avoid token limits
    if not full_text:
        full_text = abstract

    prompt = KEY_FINDINGS_PROMPT.format(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        full_text=full_text,
    )

    try:
        response = await call_llm(prompt, model=model, temperature=0.3, max_tokens=1000)

        # Parse JSON array
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        findings = json.loads(response)
        if isinstance(findings, list):
            return findings[:5]  # Max 5 findings

    except (json.JSONDecodeError, LLMError) as e:
        logger.warning(f"Failed to extract findings for {title}: {e}")

    # Fallback: return abstract as single finding if parsing fails
    if abstract:
        return [f"Study abstract: {abstract[:500]}..."]
    return ["Unable to extract key findings"]


async def analyze_methodology(
    paper: Dict[str, Any],
    parsed_doc: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> str:
    """Analyze and summarize research methodology.

    Args:
        paper: Paper metadata
        parsed_doc: Parsed PDF content
        model: LLM model to use

    Returns:
        Methodology summary text
    """
    title = paper.get("title", "Unknown")

    # Try to extract methods section
    methods_text = ""
    if parsed_doc:
        sections = parsed_doc.get("sections", [])
        for section in sections:
            section_title = section.get("title", "").lower()
            if any(term in section_title for term in ["method", "approach", "experiment", "design"]):
                methods_text = section.get("content", "")
                break

        if not methods_text:
            # Fall back to full text
            methods_text = parsed_doc.get("markdown", "")[:4000]

    if not methods_text:
        methods_text = paper.get("abstract", "")

    prompt = METHODOLOGY_ANALYSIS_PROMPT.format(
        title=title,
        methods_text=methods_text,
    )

    try:
        response = await call_llm(prompt, model=model, temperature=0.3, max_tokens=800)
        return response.strip()
    except LLMError as e:
        logger.warning(f"Failed to analyze methodology for {title}: {e}")
        return "Methodology analysis not available"


async def assess_quality(
    paper: Dict[str, Any],
    key_findings: List[str],
    parsed_doc: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Assess paper quality, identifying strengths and limitations.

    Args:
        paper: Paper metadata
        key_findings: Extracted key findings
        parsed_doc: Parsed PDF content
        model: LLM model to use

    Returns:
        Dict with "strengths" and "limitations" lists
    """
    title = paper.get("title", "Unknown")
    authors = paper.get("authors", [])
    if isinstance(authors, list):
        authors = ", ".join(authors[:5])
    year = paper.get("year", "N/A")

    content_summary = ""
    if parsed_doc:
        content_summary = parsed_doc.get("markdown", "")[:3000]
    else:
        content_summary = paper.get("abstract", "")

    findings_text = "\n".join(f"- {f}" for f in key_findings)

    prompt = QUALITY_ASSESSMENT_PROMPT.format(
        title=title,
        authors=authors,
        year=year,
        content_summary=content_summary,
        key_findings=findings_text,
    )

    try:
        response = await call_llm(prompt, model=model, temperature=0.3, max_tokens=800)

        # Parse JSON
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        result = json.loads(response)
        return {
            "strengths": result.get("strengths", [])[:4],
            "limitations": result.get("limitations", [])[:4],
        }

    except (json.JSONDecodeError, LLMError) as e:
        logger.warning(f"Failed to assess quality for {title}: {e}")
        return {
            "strengths": ["Unable to assess"],
            "limitations": ["Unable to assess"],
        }


async def analyze_single_paper(
    paper: Dict[str, Any],
    model: Optional[str] = None,
    cached_tools: Optional[CachedTools] = None,
) -> PaperSummary:
    """Perform complete critical reading analysis on a single paper.

    Supports caching for PDF acquisition and parsing.

    Args:
        paper: Paper metadata dict
        model: LLM model to use
        cached_tools: CachedTools instance for caching (optional)

    Returns:
        Complete PaperSummary
    """
    paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi", "unknown")
    title = paper.get("title", "Unknown Title")

    logger.info(f"Analyzing paper: {title}")

    # Step 1: Acquire PDF (with caching if enabled)
    pdf_path = await acquire_pdf(paper, cached_tools=cached_tools)

    # Step 2: Parse PDF (if available, with caching if enabled)
    parsed_doc = None
    if pdf_path:
        try:
            parsed_doc = await parse_paper_pdf(
                pdf_path,
                paper_id=paper_id,
                cached_tools=cached_tools,
            )
        except PDFParseError as e:
            logger.warning(f"PDF parsing failed for {title}: {e}")

    # Step 3: Extract key findings
    key_findings = await extract_key_findings(paper, parsed_doc, model)

    # Step 4: Analyze methodology
    methodology = await analyze_methodology(paper, parsed_doc, model)

    # Step 5: Assess quality
    quality = await assess_quality(paper, key_findings, parsed_doc, model)

    # Build PaperSummary
    authors = paper.get("authors", [])
    if isinstance(authors, str):
        authors = [authors]

    return PaperSummary(
        paper_id=paper_id,
        title=title,
        authors=authors[:10],  # Limit authors
        year=paper.get("year", 0),
        abstract=paper.get("abstract", "")[:1000],
        key_findings=key_findings,
        methodology=methodology,
        strengths=quality["strengths"],
        limitations=quality["limitations"],
        relevance_score=paper.get("relevance_score", 0.5),
        citations=paper.get("citations", 0),
        venue=paper.get("venue", ""),
        pdf_available=pdf_path is not None,
        source=paper.get("source", "unknown"),
    )


async def critical_reading_agent(state: LitScribeState) -> Dict[str, Any]:
    """Main entry point for the Critical Reading Agent.

    This function is called by the LangGraph workflow to analyze
    all papers selected in the discovery phase.

    Supports PDF and parse caching when cache_enabled=True.

    Args:
        state: Current workflow state

    Returns:
        State updates with analyzed papers and parsed documents
    """
    papers_to_analyze = state.get("papers_to_analyze", [])
    errors = list(state.get("errors", []))
    llm_config = state.get("llm_config", {})
    model = llm_config.get("model")
    cache_enabled = state.get("cache_enabled", True)

    if not papers_to_analyze:
        error_msg = "No papers to analyze"
        logger.warning(error_msg)
        errors.append(error_msg)
        return {
            "analyzed_papers": [],
            "parsed_documents": {},
            "errors": errors,
            "current_agent": "synthesis",  # Move to synthesis even with no papers
        }

    logger.info(f"Critical Reading Agent starting: {len(papers_to_analyze)} papers")
    logger.info(f"Cache enabled: {cache_enabled}")

    # Initialize cached tools if caching is enabled
    cached_tools = get_cached_tools(cache_enabled=cache_enabled) if cache_enabled else None

    analyzed_papers: List[PaperSummary] = []
    parsed_documents: Dict[str, Dict[str, Any]] = {}

    for i, paper in enumerate(papers_to_analyze):
        paper_id = paper.get("paper_id") or paper.get("arxiv_id") or paper.get("doi", f"paper_{i}")
        title = paper.get("title", "Unknown")

        try:
            # Analyze the paper (with caching if enabled)
            summary = await analyze_single_paper(paper, model=model, cached_tools=cached_tools)
            analyzed_papers.append(summary)

            # Store parsed document reference (if parsing was done)
            if summary["pdf_available"]:
                parsed_documents[paper_id] = {
                    "paper_id": paper_id,
                    "title": title,
                    "analyzed": True,
                }

            logger.info(f"Analyzed paper {i+1}/{len(papers_to_analyze)}: {title}")

        except Exception as e:
            error_msg = f"Failed to analyze paper '{title}': {e}"
            logger.error(error_msg)
            errors.append(error_msg)

            # Create minimal summary for failed papers
            analyzed_papers.append(PaperSummary(
                paper_id=paper_id,
                title=title,
                authors=paper.get("authors", [])[:10],
                year=paper.get("year", 0),
                abstract=paper.get("abstract", "")[:500],
                key_findings=["Analysis failed - using abstract only"],
                methodology="Not analyzed",
                strengths=[],
                limitations=["Full analysis not available"],
                relevance_score=paper.get("relevance_score", 0.3),
                citations=paper.get("citations", 0),
                venue=paper.get("venue", ""),
                pdf_available=False,
                source=paper.get("source", "unknown"),
            ))

    logger.info(f"Critical Reading complete: {len(analyzed_papers)} papers analyzed")

    return {
        "analyzed_papers": analyzed_papers,
        "parsed_documents": parsed_documents,
        "errors": errors,
        "current_agent": "synthesis",  # Next stage
    }


# Export for use in graph.py
__all__ = [
    "critical_reading_agent",
    "acquire_pdf",
    "parse_paper_pdf",
    "extract_key_findings",
    "analyze_methodology",
    "assess_quality",
    "analyze_single_paper",
]

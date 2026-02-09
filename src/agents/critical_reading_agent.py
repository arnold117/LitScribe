"""Critical Reading Agent for LitScribe.

This agent is responsible for:
1. PDF acquisition - downloading from arXiv or getting from Zotero
2. PDF parsing - extracting structured text from PDFs
3. Finding extraction - identifying key findings using LLM
4. Methodology analysis - summarizing research methods
5. Quality assessment - identifying strengths and limitations

Supports PDF and parse caching with SQLite (Phase 6.5).
Includes automatic retry with exponential backoff for LLM calls.
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")

from agents.errors import AgentError, ErrorType, LLMError, PDFNotFoundError, PDFParseError
from agents.prompts import (
    COMBINED_PAPER_ANALYSIS_PROMPT,
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
from cache.failed_papers import (
    mark_resolved_by_paper_id,
    record_failed_paper,
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 10.0  # seconds


async def retry_async(
    func: Callable[..., T],
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    **kwargs,
) -> T:
    """Execute an async function with exponential backoff retry.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        **kwargs: Keyword arguments for func

    Returns:
        Result of the function call

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")

    raise last_exception


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
    paper_id = (
        paper.get("paper_id")
        or paper.get("arxiv_id")
        or paper.get("doi")
        or "unknown"
    )

    # Check for local file first (no download needed)
    local_pdf = paper.get("local_pdf_path")
    if local_pdf and Path(local_pdf).exists():
        logger.info(f"Using local PDF for {paper_id}: {local_pdf}")
        return local_pdf

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
    tracker=None,
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
        response = await call_llm(prompt, model=model, temperature=0.3, max_tokens=1000, tracker=tracker, agent_name="critical_reading")

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

    except Exception as e:
        logger.warning(f"Failed to extract findings for {title}: {e}")

    # Fallback: return abstract as single finding if parsing fails
    if abstract:
        return [f"Study abstract: {abstract[:500]}..."]
    return ["Unable to extract key findings"]


async def analyze_methodology(
    paper: Dict[str, Any],
    parsed_doc: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    tracker=None,
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
        response = await call_llm(prompt, model=model, temperature=0.3, max_tokens=800, tracker=tracker, agent_name="critical_reading")
        return response.strip()
    except Exception as e:
        logger.warning(f"Failed to analyze methodology for {title}: {e}")
        return "Methodology analysis not available"


async def assess_quality(
    paper: Dict[str, Any],
    key_findings: List[str],
    parsed_doc: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    tracker=None,
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
        response = await call_llm(prompt, model=model, temperature=0.3, max_tokens=800, tracker=tracker, agent_name="critical_reading")

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

    except Exception as e:
        logger.warning(f"Failed to assess quality for {title}: {e}")
        return {
            "strengths": ["Unable to assess"],
            "limitations": ["Unable to assess"],
        }


async def analyze_paper_combined(
    paper: Dict[str, Any],
    parsed_doc: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    research_question: str = "",
    tracker=None,
) -> Dict[str, Any]:
    """Perform combined analysis using a single LLM call.

    This is faster than 3 separate calls (findings, methodology, quality).

    Args:
        paper: Paper metadata with abstract
        parsed_doc: Parsed PDF content (optional)
        model: LLM model to use
        research_question: Research question for context-aware analysis

    Returns:
        Dict with key_findings, methodology, strengths, limitations, relevance_to_question
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
        full_text = parsed_doc.get("markdown", "")[:8000]
    if not full_text:
        full_text = abstract

    prompt = COMBINED_PAPER_ANALYSIS_PROMPT.format(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        full_text=full_text,
        research_question=research_question or "General academic review",
    )

    try:
        response = await call_llm(prompt, model=model, temperature=0.3, max_tokens=2500, tracker=tracker, agent_name="critical_reading")

        # Parse JSON response
        response = response.strip()
        if response.startswith("```"):
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    response = part
                    break

        result = json.loads(response)

        return {
            "key_findings": result.get("key_findings", [])[:8],
            "methodology": result.get("methodology", "Methodology not analyzed"),
            "strengths": result.get("strengths", [])[:4],
            "limitations": result.get("limitations", [])[:4],
            "relevance_to_question": float(result.get("relevance_to_question", 0.5)),
        }

    except Exception as e:
        logger.warning(f"Combined analysis failed for {title}: {e}")
        # Return defaults
        return {
            "key_findings": [f"Study abstract: {abstract[:500]}..."] if abstract else ["Unable to extract findings"],
            "methodology": "Methodology analysis not available",
            "strengths": ["Unable to assess"],
            "limitations": ["Unable to assess"],
            "relevance_to_question": 0.5,
        }


async def analyze_single_paper(
    paper: Dict[str, Any],
    model: Optional[str] = None,
    cached_tools: Optional[CachedTools] = None,
    use_combined_prompt: bool = True,
    research_question: str = "",
    tracker=None,
) -> PaperSummary:
    """Perform complete critical reading analysis on a single paper.

    Supports caching for PDF acquisition and parsing.
    Uses combined prompt by default for faster analysis (1 LLM call vs 3).

    Args:
        paper: Paper metadata dict
        model: LLM model to use
        cached_tools: CachedTools instance for caching (optional)
        use_combined_prompt: Use single combined LLM call (faster)
        research_question: Research question for context-aware analysis

    Returns:
        Complete PaperSummary
    """
    paper_id = (
        paper.get("paper_id")
        or paper.get("arxiv_id")
        or paper.get("doi")
        or "unknown"
    )
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

    # Step 3: Analyze paper (combined or separate)
    if use_combined_prompt:
        # Single LLM call for all analysis
        analysis = await analyze_paper_combined(
            paper, parsed_doc, model, research_question=research_question, tracker=tracker,
        )
        key_findings = analysis["key_findings"]
        methodology = analysis["methodology"]
        quality = {"strengths": analysis["strengths"], "limitations": analysis["limitations"]}
        # Update relevance_score from LLM assessment if available
        llm_relevance = analysis.get("relevance_to_question")
        if llm_relevance is not None:
            relevance_score = llm_relevance
        else:
            relevance_score = paper.get("relevance_score", 0.5)
    else:
        # Original 3 separate calls
        key_findings = await extract_key_findings(paper, parsed_doc, model, tracker=tracker)
        methodology = await analyze_methodology(paper, parsed_doc, model, tracker=tracker)
        quality = await assess_quality(paper, key_findings, parsed_doc, model, tracker=tracker)
        relevance_score = paper.get("relevance_score", 0.5)

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
        relevance_score=relevance_score,
        citations=paper.get("citations", 0),
        venue=paper.get("venue", ""),
        pdf_available=pdf_path is not None,
        source=paper.get("source", "unknown"),
    )


def _local_file_to_paper(file_path: str) -> Dict[str, Any]:
    """Convert a local PDF file path to a paper dict for the pipeline.

    Args:
        file_path: Absolute path to a local PDF file

    Returns:
        Paper dict with paper_id derived from filename hash
    """
    path = Path(file_path)
    file_hash = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:12]
    paper_id = f"local:{file_hash}"

    return {
        "paper_id": paper_id,
        "title": path.stem.replace("_", " ").replace("-", " "),
        "authors": [],
        "abstract": "",
        "year": 0,
        "venue": "",
        "citations": 0,
        "source": "local_file",
        "local_pdf_path": str(path.resolve()),
        "search_origin": "local_file",
    }


async def critical_reading_agent(state: LitScribeState) -> Dict[str, Any]:
    """Main entry point for the Critical Reading Agent.

    This function is called by the LangGraph workflow to analyze
    all papers selected in the discovery phase.

    Supports PDF and parse caching when cache_enabled=True.
    Uses parallel processing for faster analysis.
    Handles local PDF files specified via local_files state field.

    Args:
        state: Current workflow state

    Returns:
        State updates with analyzed papers and parsed documents
    """
    papers_to_analyze = list(state.get("papers_to_analyze", []))
    local_files = state.get("local_files", [])
    errors = list(state.get("errors", []))

    # Inject local files as papers (skip PDF download for these)
    for file_path in local_files:
        path = Path(file_path)
        if path.exists() and path.suffix.lower() == ".pdf":
            local_paper = _local_file_to_paper(file_path)
            papers_to_analyze.append(local_paper)
            logger.info(f"Added local file: {path.name}")
        else:
            error_msg = f"Local file not found or not PDF: {file_path}"
            logger.warning(error_msg)
            errors.append(error_msg)
    llm_config = state.get("llm_config", {})
    model = llm_config.get("model")
    cache_enabled = state.get("cache_enabled", True)
    research_question = state.get("research_question", "")
    max_concurrent = state.get("max_concurrent", 3)  # Limit concurrent LLM calls
    tracker = state.get("token_tracker")

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
    logger.info(f"Cache enabled: {cache_enabled}, Max concurrent: {max_concurrent}")

    # Initialize cached tools if caching is enabled
    cached_tools = get_cached_tools(cache_enabled=cache_enabled) if cache_enabled else None

    # Use semaphore to limit concurrent analysis
    semaphore = asyncio.Semaphore(max_concurrent)

    async def analyze_with_fallback(i: int, paper: Dict[str, Any]) -> PaperSummary:
        """Analyze a single paper with error handling and fallback."""
        async with semaphore:
            paper_id = (
                paper.get("paper_id")
                or paper.get("arxiv_id")
                or paper.get("doi")
                or f"paper_{i}"
            )
            title = paper.get("title", "Unknown")

            try:
                # Analyze the paper (with caching if enabled, with retry)
                summary = await retry_async(
                    analyze_single_paper,
                    paper,
                    model=model,
                    cached_tools=cached_tools,
                    use_combined_prompt=True,  # Use optimized combined prompt
                    research_question=research_question,
                    tracker=tracker,
                    max_retries=MAX_RETRIES,
                )

                # Mark any previous failures as resolved
                if cache_enabled:
                    try:
                        await mark_resolved_by_paper_id(paper_id, resolution="success")
                    except Exception:
                        pass

                logger.info(f"Analyzed paper {i+1}/{len(papers_to_analyze)}: {title}")
                return summary

            except Exception as e:
                error_msg = f"Failed to analyze paper '{title}': {e}"
                logger.error(error_msg)
                errors.append(error_msg)

                # Record failure to retry queue (non-blocking)
                if cache_enabled:
                    try:
                        await record_failed_paper(
                            paper_id=paper_id,
                            title=title,
                            error_message=str(e),
                            paper_data=paper,
                            research_question=research_question,
                            max_retries=MAX_RETRIES,
                        )
                    except Exception as queue_err:
                        logger.warning(f"Failed to record to retry queue: {queue_err}")

                # Return minimal summary for failed papers
                return _create_fallback_summary(i, paper, paper_id, title)

    def _create_fallback_summary(
        i: int, paper: Dict[str, Any], paper_id: str, title: str
    ) -> PaperSummary:
        """Create minimal summary for failed papers."""
        try:
            authors_raw = paper.get("authors")
            if authors_raw is None:
                authors = []
            elif isinstance(authors_raw, str):
                authors = [authors_raw]
            else:
                authors = list(authors_raw)[:10]

            abstract_raw = paper.get("abstract")
            abstract = str(abstract_raw)[:500] if abstract_raw else ""

            return PaperSummary(
                paper_id=paper_id,
                title=title or "Unknown",
                authors=authors,
                year=paper.get("year") or 0,
                abstract=abstract,
                key_findings=["Analysis failed - using abstract only"],
                methodology="Not analyzed",
                strengths=[],
                limitations=["Full analysis not available"],
                relevance_score=paper.get("relevance_score") or 0.3,
                citations=paper.get("citations") or 0,
                venue=paper.get("venue") or "",
                pdf_available=False,
                source=paper.get("source") or "unknown",
            )
        except Exception as inner_e:
            logger.error(f"Even minimal summary failed for '{title}': {inner_e}")
            return PaperSummary(
                paper_id=paper_id or f"unknown_{i}",
                title=str(title) if title else "Unknown",
                authors=[],
                year=0,
                abstract="",
                key_findings=["Analysis failed"],
                methodology="Not analyzed",
                strengths=[],
                limitations=["Analysis not available"],
                relevance_score=0.3,
                citations=0,
                venue="",
                pdf_available=False,
                source="unknown",
            )

    # Analyze all papers in parallel (with semaphore limiting concurrency)
    logger.info(f"Starting parallel analysis of {len(papers_to_analyze)} papers...")
    analyzed_papers = await asyncio.gather(
        *[analyze_with_fallback(i, paper) for i, paper in enumerate(papers_to_analyze)],
        return_exceptions=False,  # Exceptions handled inside analyze_with_fallback
    )

    # Build parsed_documents dict
    parsed_documents: Dict[str, Dict[str, Any]] = {}
    for summary in analyzed_papers:
        if summary["pdf_available"]:
            parsed_documents[summary["paper_id"]] = {
                "paper_id": summary["paper_id"],
                "title": summary["title"],
                "analyzed": True,
            }

    logger.info(f"Critical Reading complete: {len(analyzed_papers)} papers analyzed")

    return {
        "analyzed_papers": list(analyzed_papers),
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

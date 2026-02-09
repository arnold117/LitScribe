"""PubMed MCP Server - Search and retrieve biomedical literature from PubMed/NCBI."""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Bio import Entrez, Medline

from models.pubmed_article import PubMedArticle
from utils.config import Config

# Configure Entrez
Entrez.email = Config.NCBI_EMAIL
if Config.NCBI_API_KEY:
    Entrez.api_key = Config.NCBI_API_KEY


def _parse_articles(handle) -> List[PubMedArticle]:
    """Parse Medline records from Entrez handle."""
    records = list(Medline.parse(handle))
    handle.close()
    return [PubMedArticle.from_medline_record(r) for r in records]


async def search_pubmed(
    query: str,
    max_results: int = 20,
    sort: str = "relevance",
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> dict:
    """
    Search PubMed for articles matching the query.

    Args:
        query: Search query (supports PubMed search syntax)
        max_results: Maximum number of results to return (default: 20, max: 100)
        sort: Sort order - "relevance" or "date" (default: relevance)
        min_date: Minimum publication date (YYYY/MM/DD or YYYY)
        max_date: Maximum publication date (YYYY/MM/DD or YYYY)

    Returns:
        Dictionary with total count and list of articles
    """
    max_results = min(max_results, 100)
    sort_order = "relevance" if sort == "relevance" else "pub_date"

    # Build search parameters
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": sort_order,
    }

    if min_date:
        search_params["mindate"] = min_date
    if max_date:
        search_params["maxdate"] = max_date
    if min_date or max_date:
        search_params["datetype"] = "pdat"

    # Run search in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    def do_search():
        handle = Entrez.esearch(**search_params)
        record = Entrez.read(handle)
        handle.close()
        return record

    record = await loop.run_in_executor(None, do_search)

    total_count = int(record.get("Count", 0))
    id_list = record.get("IdList", [])

    if not id_list:
        return {"total_count": total_count, "articles": []}

    # Fetch article details
    def do_fetch():
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(id_list),
            rettype="medline",
            retmode="text",
        )
        return _parse_articles(handle)

    articles = await loop.run_in_executor(None, do_fetch)

    return {
        "total_count": total_count,
        "returned_count": len(articles),
        "articles": [a.to_dict() for a in articles],
    }


async def get_article_details(pmid: str) -> dict:
    """
    Get detailed information for a specific PubMed article.

    Args:
        pmid: PubMed ID of the article

    Returns:
        Article details including abstract, MeSH terms, and metadata
    """
    loop = asyncio.get_event_loop()

    def do_fetch():
        handle = Entrez.efetch(
            db="pubmed",
            id=pmid,
            rettype="medline",
            retmode="text",
        )
        articles = _parse_articles(handle)
        return articles[0] if articles else None

    article = await loop.run_in_executor(None, do_fetch)

    if not article:
        return {"error": f"Article with PMID {pmid} not found"}

    return article.to_dict()


async def get_related_articles(pmid: str, max_results: int = 10) -> dict:
    """
    Get articles related to a specific PubMed article.

    Args:
        pmid: PubMed ID of the source article
        max_results: Maximum number of related articles (default: 10)

    Returns:
        List of related articles
    """
    max_results = min(max_results, 50)
    loop = asyncio.get_event_loop()

    def do_link():
        # Use elink to find related articles
        handle = Entrez.elink(
            dbfrom="pubmed",
            db="pubmed",
            id=pmid,
            linkname="pubmed_pubmed",
        )
        record = Entrez.read(handle)
        handle.close()
        return record

    record = await loop.run_in_executor(None, do_link)

    # Extract related PMIDs
    related_ids = []
    if record and record[0].get("LinkSetDb"):
        for link_set in record[0]["LinkSetDb"]:
            if link_set.get("LinkName") == "pubmed_pubmed":
                for link in link_set.get("Link", [])[:max_results]:
                    related_ids.append(link["Id"])
                break

    if not related_ids:
        return {"source_pmid": pmid, "related_articles": []}

    # Fetch related article details
    def do_fetch():
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(related_ids),
            rettype="medline",
            retmode="text",
        )
        return _parse_articles(handle)

    articles = await loop.run_in_executor(None, do_fetch)

    return {
        "source_pmid": pmid,
        "related_count": len(articles),
        "related_articles": [a.to_dict() for a in articles],
    }


async def search_mesh_terms(term: str, max_results: int = 10) -> dict:
    """
    Search for MeSH (Medical Subject Headings) terms.

    Args:
        term: Term to search for
        max_results: Maximum number of results (default: 10)

    Returns:
        List of matching MeSH terms with their tree numbers
    """
    loop = asyncio.get_event_loop()

    def do_search():
        # Search MeSH database
        handle = Entrez.esearch(
            db="mesh",
            term=term,
            retmax=max_results,
        )
        record = Entrez.read(handle)
        handle.close()
        return record

    record = await loop.run_in_executor(None, do_search)

    id_list = record.get("IdList", [])
    if not id_list:
        return {"query": term, "mesh_terms": []}

    # Fetch MeSH term details
    def do_fetch():
        handle = Entrez.efetch(
            db="mesh",
            id=",".join(id_list),
            rettype="full",
            retmode="text",
        )
        content = handle.read()
        handle.close()
        return content

    content = await loop.run_in_executor(None, do_fetch)

    # Parse MeSH records (simplified parsing)
    mesh_terms = []
    current_term = {}

    for line in content.strip().split("\n"):
        if line.startswith("*NEWRECORD"):
            if current_term:
                mesh_terms.append(current_term)
            current_term = {}
        elif line.startswith("DESCRIPTORNAME"):
            current_term["name"] = line.split("=", 1)[1].strip() if "=" in line else ""
        elif line.startswith("MESHHEADINGLIST"):
            current_term["heading"] = line.split("=", 1)[1].strip() if "=" in line else ""
        elif line.startswith("TREELOC"):
            if "tree_numbers" not in current_term:
                current_term["tree_numbers"] = []
            current_term["tree_numbers"].append(line.split("=", 1)[1].strip() if "=" in line else "")
        elif line.startswith("SCOPE"):
            current_term["scope_note"] = line.split("=", 1)[1].strip() if "=" in line else ""

    if current_term:
        mesh_terms.append(current_term)

    return {
        "query": term,
        "count": len(mesh_terms),
        "mesh_terms": mesh_terms,
    }


async def get_citations(pmid: str, direction: str = "both") -> dict:
    """
    Get citation information for an article.

    Args:
        pmid: PubMed ID of the article
        direction: "cited_by" (papers citing this), "references" (papers this cites), or "both"

    Returns:
        Citation information including cited_by and references lists
    """
    loop = asyncio.get_event_loop()
    result = {"pmid": pmid}

    async def get_cited_by():
        """Get papers that cite this article."""
        def do_link():
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pubmed",
                id=pmid,
                linkname="pubmed_pubmed_citedin",
            )
            record = Entrez.read(handle)
            handle.close()
            return record

        record = await loop.run_in_executor(None, do_link)

        citing_ids = []
        if record and record[0].get("LinkSetDb"):
            for link_set in record[0]["LinkSetDb"]:
                if link_set.get("LinkName") == "pubmed_pubmed_citedin":
                    for link in link_set.get("Link", []):
                        citing_ids.append(link["Id"])
                    break

        return citing_ids

    async def get_references():
        """Get papers that this article cites."""
        def do_link():
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pubmed",
                id=pmid,
                linkname="pubmed_pubmed_refs",
            )
            record = Entrez.read(handle)
            handle.close()
            return record

        record = await loop.run_in_executor(None, do_link)

        ref_ids = []
        if record and record[0].get("LinkSetDb"):
            for link_set in record[0]["LinkSetDb"]:
                if link_set.get("LinkName") == "pubmed_pubmed_refs":
                    for link in link_set.get("Link", []):
                        ref_ids.append(link["Id"])
                    break

        return ref_ids

    # Fetch citations based on direction
    if direction in ("cited_by", "both"):
        result["cited_by"] = await get_cited_by()
        result["cited_by_count"] = len(result["cited_by"])

    if direction in ("references", "both"):
        result["references"] = await get_references()
        result["references_count"] = len(result["references"])

    return result


async def batch_get_articles(pmids: List[str]) -> dict:
    """
    Get details for multiple PubMed articles at once.

    Args:
        pmids: List of PubMed IDs (max 100)

    Returns:
        List of article details
    """
    pmids = pmids[:100]  # Limit to 100
    loop = asyncio.get_event_loop()

    def do_fetch():
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(pmids),
            rettype="medline",
            retmode="text",
        )
        return _parse_articles(handle)

    articles = await loop.run_in_executor(None, do_fetch)

    return {
        "requested_count": len(pmids),
        "returned_count": len(articles),
        "articles": [a.to_dict() for a in articles],
    }



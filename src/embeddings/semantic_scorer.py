"""Semantic relevance scoring using sentence-transformer embeddings.

Reuses the all-MiniLM-L6-v2 model from graphrag.entity_linker for
consistency and to avoid loading duplicate models.
"""

import logging
import re
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level caches
_query_embedding_cache: Dict[str, np.ndarray] = {}


def get_embedding_model():
    """Get the shared sentence-transformer model (singleton from entity_linker)."""
    from graphrag.entity_linker import get_embedding_model as _get_model

    return _get_model()


def compute_query_embedding(text: str) -> np.ndarray:
    """Compute and cache a normalized embedding for a query string."""
    if text in _query_embedding_cache:
        return _query_embedding_cache[text]

    model = get_embedding_model()
    embedding = model.encode([text], show_progress_bar=False)[0]

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    _query_embedding_cache[text] = embedding
    return embedding


def compute_paper_embeddings_batch(texts: List[str]) -> np.ndarray:
    """Batch-encode paper texts into normalized embeddings.

    Args:
        texts: List of "title. abstract" strings

    Returns:
        Array of shape (n, embedding_dim), L2-normalized per row.
    """
    if not texts:
        return np.array([])

    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return embeddings / norms


def compute_semantic_relevance(
    papers: list,
    research_question: str,
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7,
) -> None:
    """Compute blended relevance scores for papers (modifies in place).

    Score = semantic_weight * cosine_sim(question, title+abstract)
          + keyword_weight * keyword_score

    Papers without meaningful abstracts fall back to keyword-only scoring.

    Args:
        papers: List of UnifiedPaper objects (modified in place)
        research_question: The original research question
        keyword_weight: Weight for keyword-based score (default 0.3)
        semantic_weight: Weight for semantic similarity (default 0.7)
    """
    if not papers:
        return

    # Step 1: Compute keyword scores first (as fallback and secondary signal)
    from aggregators.unified_search import _compute_keyword_relevance

    _compute_keyword_relevance(papers, research_question)
    keyword_scores = {id(p): p.relevance_score for p in papers}

    # Step 2: Build paper text representations
    texts = []
    has_text = []
    for paper in papers:
        title = (paper.title or "").strip()
        abstract = (paper.abstract or "").strip()
        if abstract and len(abstract) > 30:
            texts.append(f"{title}. {abstract[:500]}")
            has_text.append(True)
        elif title:
            texts.append(title)
            has_text.append(True)
        else:
            texts.append("")
            has_text.append(False)

    # Step 3: Compute embeddings
    try:
        query_embedding = compute_query_embedding(research_question)
        paper_embeddings = compute_paper_embeddings_batch(texts)

        if paper_embeddings.size == 0:
            return  # Keep keyword scores as-is

        # Step 4: Cosine similarity (already normalized)
        similarities = paper_embeddings @ query_embedding

        # Detect pure CJK query — all-MiniLM-L6-v2 is English-only, so
        # cross-language similarity will be near 0 and must not kill papers.
        _cjk_query = bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', research_question))

        # Step 5: Blend scores
        for i, paper in enumerate(papers):
            kw_score = keyword_scores[id(paper)]

            if has_text[i] and i < len(similarities):
                sem_score = float(max(0.0, min(1.0, similarities[i])))
                blended = semantic_weight * sem_score + keyword_weight * kw_score
                # CJK cross-language floor: English-only embedding model gives
                # near-zero similarity for CJK queries vs English papers.
                # Preserve a neutral floor so LLM selection decides relevance.
                if _cjk_query and blended < 0.35:
                    blended = max(blended, 0.35)
                paper.relevance_score = blended
            else:
                # No text to embed — keep keyword score only
                paper.relevance_score = kw_score

    except Exception as e:
        logger.warning(f"Semantic scoring failed, falling back to keyword-only: {e}")
        # Keyword scores already set in step 1, nothing to do


def deduplicate_queries_by_similarity(
    queries: List[str],
    threshold: float = 0.85,
) -> List[str]:
    """Remove near-duplicate queries using cosine similarity.

    Args:
        queries: List of search query strings
        threshold: Similarity above which a query is considered duplicate

    Returns:
        Deduplicated list (order preserved, later duplicates removed)
    """
    if len(queries) <= 1:
        return queries

    try:
        model = get_embedding_model()
        embeddings = model.encode(queries, show_progress_bar=False)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = embeddings / norms

        keep = [True] * len(queries)
        for i in range(len(queries)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(queries)):
                if not keep[j]:
                    continue
                sim = float(np.dot(normalized[i], normalized[j]))
                if sim >= threshold:
                    keep[j] = False
                    logger.debug(
                        f"Query dedup: dropped '{queries[j]}' "
                        f"(sim={sim:.2f} with '{queries[i]}')"
                    )

        result = [q for q, k in zip(queries, keep) if k]
        if len(result) < len(queries):
            logger.info(
                f"Query dedup: {len(queries)} -> {len(result)} queries "
                f"(threshold={threshold})"
            )
        return result

    except Exception as e:
        logger.warning(f"Query dedup failed, returning all queries: {e}")
        return queries

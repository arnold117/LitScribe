"""Entity extraction from academic papers using LLM.

This module extracts structured entities (methods, datasets, metrics, concepts)
from paper summaries and parsed documents.
"""

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agents.state import EntityMention, ExtractedEntity, PaperSummary
from graphrag.prompts import (
    ENTITY_EXTRACTION_CONTENT_TEMPLATE,
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_EXTRACTION_SYSTEM,
)

logger = logging.getLogger(__name__)


def _clean_json_response(response: str) -> str:
    """Clean LLM response to extract valid JSON.

    Handles common issues like markdown code blocks, leading/trailing text.

    Args:
        response: Raw LLM response

    Returns:
        Cleaned JSON string, or empty string if no JSON found
    """
    if not response:
        return ""

    text = response.strip()

    # Handle markdown code blocks: ```json ... ``` or ``` ... ```
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            # Skip empty parts
            if not part:
                continue
            # Remove optional language identifier
            if part.startswith("json"):
                part = part[4:].strip()
            # Check if it looks like JSON
            if part.startswith("{") or part.startswith("["):
                text = part
                break

    # Find JSON object boundaries
    start_idx = -1
    end_idx = -1

    # Try to find JSON object
    for i, char in enumerate(text):
        if char == "{":
            start_idx = i
            break

    if start_idx >= 0:
        # Find matching closing brace
        depth = 0
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

    if start_idx >= 0 and end_idx > start_idx:
        return text[start_idx:end_idx]

    return text if text.startswith("{") else ""


def generate_entity_id(name: str, entity_type: str) -> str:
    """Generate a unique entity ID from name and type.

    Args:
        name: Canonical entity name
        entity_type: Entity type (method, dataset, metric, concept)

    Returns:
        Unique entity ID (hash of normalized name + type)
    """
    normalized = f"{name.lower().strip()}:{entity_type.lower()}"
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for consistent matching.

    Args:
        name: Raw entity name

    Returns:
        Normalized name
    """
    # Remove common suffixes/prefixes
    name = name.strip()

    # Handle common variations
    replacements = {
        " model": "",
        " algorithm": "",
        " method": "",
        " architecture": "",
        " framework": "",
        " dataset": "",
        " benchmark": "",
        " corpus": "",
    }

    name_lower = name.lower()
    for old, new in replacements.items():
        if name_lower.endswith(old):
            name = name[: -len(old)]
            break

    return name.strip()


async def extract_entities_from_paper(
    paper: PaperSummary,
    parsed_doc: Optional[Dict[str, Any]] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    research_question: str = "",
) -> Tuple[List[ExtractedEntity], List[EntityMention]]:
    """Extract entities from a single paper using LLM.

    Args:
        paper: Paper summary with metadata
        parsed_doc: Optional parsed document with full text
        llm_config: LLM configuration
        research_question: Research question for contextual extraction

    Returns:
        Tuple of (extracted entities, entity mentions)
    """
    from agents.tools import call_llm_with_system

    # Build content section if we have parsed doc
    content_section = ""
    if parsed_doc and parsed_doc.get("markdown"):
        content = parsed_doc["markdown"][:3000]  # Limit to 3000 chars
        content_section = ENTITY_EXTRACTION_CONTENT_TEMPLATE.format(content=content)

    # Format the prompt
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        title=paper.get("title", "Unknown"),
        authors=", ".join(paper.get("authors", [])[:3]),  # First 3 authors
        year=paper.get("year", "Unknown"),
        abstract=paper.get("abstract", "")[:1000],  # Limit abstract
        content_section=content_section,
        research_question=research_question or "General academic research",
    )

    # Call LLM with system prompt
    try:
        response = await call_llm_with_system(
            system_prompt=ENTITY_EXTRACTION_SYSTEM,
            user_prompt=prompt,
            model=llm_config.get("model") if llm_config else None,
            temperature=0.3,
            max_tokens=2000,
        )

        # Clean and parse response
        cleaned = _clean_json_response(response)
        if not cleaned:
            logger.warning(f"Empty response for entity extraction")
            raw_entities = []
        else:
            result = json.loads(cleaned)
            raw_entities = result.get("entities", [])

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse entity extraction response: {e}")
        logger.debug(f"Raw response was: {response[:200] if response else 'None'}...")
        raw_entities = []
    except Exception as e:
        logger.error(f"Entity extraction failed for {paper.get('paper_id')}: {e}")
        raw_entities = []

    # Convert to ExtractedEntity and EntityMention
    entities = []
    mentions = []
    paper_id = paper.get("paper_id", "unknown")

    for raw in raw_entities:
        name = raw.get("name", "").strip()
        entity_type = raw.get("type", "concept").lower()

        if not name:
            continue

        # Validate entity type
        if entity_type not in ("method", "dataset", "metric", "concept"):
            entity_type = "concept"

        # Normalize name
        normalized_name = normalize_entity_name(name)
        entity_id = generate_entity_id(normalized_name, entity_type)

        # Create entity
        entity = ExtractedEntity(
            entity_id=entity_id,
            name=normalized_name,
            entity_type=entity_type,
            aliases=[name] + raw.get("aliases", []),
            description=raw.get("description", ""),
            paper_ids=[paper_id],
            frequency=1,
        )
        entities.append(entity)

        # Create mention
        mention = EntityMention(
            entity_id=entity_id,
            paper_id=paper_id,
            context=raw.get("description", "")[:200],
            section="abstract" if not parsed_doc else "full_text",
            confidence=0.9,  # High confidence for LLM extraction
        )
        mentions.append(mention)

    logger.info(f"Extracted {len(entities)} entities from paper {paper_id}")
    return entities, mentions


async def extract_entities_batch(
    papers: List[PaperSummary],
    parsed_documents: Optional[Dict[str, Dict[str, Any]]] = None,
    batch_size: int = 10,
    max_concurrent: int = 5,
    llm_config: Optional[Dict[str, Any]] = None,
    research_question: str = "",
) -> Tuple[List[ExtractedEntity], List[EntityMention]]:
    """Extract entities from multiple papers with batching.

    Args:
        papers: List of paper summaries
        parsed_documents: Dict of paper_id -> parsed document
        batch_size: Number of papers to process in each batch
        max_concurrent: Maximum concurrent LLM calls
        llm_config: LLM configuration
        research_question: Research question for contextual extraction

    Returns:
        Tuple of (all entities, all mentions)
    """
    if parsed_documents is None:
        parsed_documents = {}

    all_entities: Dict[str, ExtractedEntity] = {}  # entity_id -> entity
    all_mentions: List[EntityMention] = []

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_paper(paper: PaperSummary):
        async with semaphore:
            paper_id = paper.get("paper_id", "unknown")
            parsed_doc = parsed_documents.get(paper_id)
            return await extract_entities_from_paper(
                paper, parsed_doc, llm_config, research_question=research_question,
            )

    # Process in batches
    for batch_start in range(0, len(papers), batch_size):
        batch = papers[batch_start : batch_start + batch_size]
        logger.info(
            f"Processing entity extraction batch {batch_start // batch_size + 1}"
            f" ({len(batch)} papers)"
        )

        # Process batch concurrently
        results = await asyncio.gather(
            *[process_paper(paper) for paper in batch],
            return_exceptions=True,
        )

        # Merge results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch extraction error: {result}")
                continue

            entities, mentions = result

            # Merge entities (combine paper_ids and frequency)
            for entity in entities:
                eid = entity["entity_id"]
                if eid in all_entities:
                    # Merge with existing entity
                    existing = all_entities[eid]
                    existing["paper_ids"] = list(
                        set(existing["paper_ids"]) | set(entity["paper_ids"])
                    )
                    existing["frequency"] += 1
                    # Merge aliases
                    existing["aliases"] = list(
                        set(existing["aliases"]) | set(entity["aliases"])
                    )
                else:
                    all_entities[eid] = entity

            all_mentions.extend(mentions)

    logger.info(
        f"Entity extraction complete: {len(all_entities)} unique entities, "
        f"{len(all_mentions)} mentions"
    )

    return list(all_entities.values()), all_mentions


def merge_entities(
    entities: List[ExtractedEntity],
) -> Dict[str, ExtractedEntity]:
    """Merge duplicate entities by ID.

    Args:
        entities: List of entities (may have duplicates)

    Returns:
        Dict of entity_id -> merged entity
    """
    merged: Dict[str, ExtractedEntity] = {}

    for entity in entities:
        eid = entity["entity_id"]
        if eid in merged:
            existing = merged[eid]
            existing["paper_ids"] = list(
                set(existing["paper_ids"]) | set(entity["paper_ids"])
            )
            existing["frequency"] += entity["frequency"]
            existing["aliases"] = list(
                set(existing["aliases"]) | set(entity["aliases"])
            )
        else:
            merged[eid] = entity.copy()

    return merged

"""GraphRAG cache for persistent entity and community storage.

This module provides caching for GraphRAG data (entities, mentions,
edges, communities) in SQLite for reuse across runs.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import aiosqlite

from agents.state import Community, EntityMention, ExtractedEntity, GraphEdge
from cache.database import get_cache_db

logger = logging.getLogger(__name__)


class GraphRAGCache:
    """Cache for GraphRAG entities, mentions, and communities."""

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._db = get_cache_db() if cache_enabled else None

    async def _ensure_initialized(self) -> None:
        """Ensure database is initialized."""
        if self._db:
            await self._db.init_schema_async()

    # =========================================================================
    # Entity Operations
    # =========================================================================

    async def get_entities_for_papers(
        self, paper_ids: List[str]
    ) -> Tuple[List[ExtractedEntity], List[EntityMention]]:
        """Get cached entities and mentions for given papers.

        Args:
            paper_ids: List of paper IDs to look up

        Returns:
            Tuple of (entities, mentions) that are already cached
        """
        if not self.cache_enabled or not paper_ids:
            return [], []

        await self._ensure_initialized()

        entities: Dict[str, ExtractedEntity] = {}
        mentions: List[EntityMention] = []

        async with aiosqlite.connect(self._db.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Get mentions for these papers
            placeholders = ",".join("?" * len(paper_ids))
            cursor = await db.execute(
                f"""
                SELECT entity_id, paper_id, context, section, confidence
                FROM entity_mentions
                WHERE paper_id IN ({placeholders})
                """,
                paper_ids,
            )
            rows = await cursor.fetchall()

            entity_ids = set()
            for row in rows:
                mentions.append(
                    EntityMention(
                        entity_id=row["entity_id"],
                        paper_id=row["paper_id"],
                        context=row["context"] or "",
                        section=row["section"] or "",
                        confidence=row["confidence"] or 0.9,
                    )
                )
                entity_ids.add(row["entity_id"])

            # Get entities
            if entity_ids:
                placeholders = ",".join("?" * len(entity_ids))
                cursor = await db.execute(
                    f"""
                    SELECT entity_id, name, entity_type, aliases, description,
                           frequency, paper_ids
                    FROM entities
                    WHERE entity_id IN ({placeholders})
                    """,
                    list(entity_ids),
                )
                rows = await cursor.fetchall()

                for row in rows:
                    entities[row["entity_id"]] = ExtractedEntity(
                        entity_id=row["entity_id"],
                        name=row["name"],
                        entity_type=row["entity_type"],
                        aliases=json.loads(row["aliases"]) if row["aliases"] else [],
                        description=row["description"] or "",
                        frequency=row["frequency"] or 1,
                        paper_ids=json.loads(row["paper_ids"]) if row["paper_ids"] else [],
                    )

        logger.info(
            f"Retrieved {len(entities)} cached entities, {len(mentions)} mentions"
        )
        return list(entities.values()), mentions

    async def get_papers_with_entities(self) -> Set[str]:
        """Get set of paper IDs that already have extracted entities.

        Returns:
            Set of paper IDs with cached entities
        """
        if not self.cache_enabled:
            return set()

        await self._ensure_initialized()

        async with aiosqlite.connect(self._db.db_path) as db:
            cursor = await db.execute(
                "SELECT DISTINCT paper_id FROM entity_mentions"
            )
            rows = await cursor.fetchall()
            return {row[0] for row in rows}

    async def save_entities(
        self,
        entities: List[ExtractedEntity],
        mentions: List[EntityMention],
    ) -> None:
        """Save entities and mentions to cache.

        Args:
            entities: List of extracted entities
            mentions: List of entity mentions
        """
        if not self.cache_enabled:
            return

        await self._ensure_initialized()

        async with aiosqlite.connect(self._db.db_path) as db:
            # Save entities (upsert)
            for entity in entities:
                await db.execute(
                    """
                    INSERT INTO entities (entity_id, name, entity_type, aliases,
                                         description, frequency, paper_ids, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_id) DO UPDATE SET
                        frequency = frequency + excluded.frequency,
                        paper_ids = excluded.paper_ids,
                        aliases = excluded.aliases,
                        updated_at = excluded.updated_at
                    """,
                    (
                        entity["entity_id"],
                        entity["name"],
                        entity["entity_type"],
                        json.dumps(entity.get("aliases", [])),
                        entity.get("description", ""),
                        entity.get("frequency", 1),
                        json.dumps(entity.get("paper_ids", [])),
                        datetime.now().isoformat(),
                    ),
                )

            # Save mentions (insert or ignore duplicates)
            for mention in mentions:
                await db.execute(
                    """
                    INSERT OR IGNORE INTO entity_mentions
                        (entity_id, paper_id, context, section, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        mention["entity_id"],
                        mention["paper_id"],
                        mention.get("context", ""),
                        mention.get("section", ""),
                        mention.get("confidence", 0.9),
                    ),
                )

            await db.commit()

        logger.info(f"Saved {len(entities)} entities, {len(mentions)} mentions to cache")

    async def get_all_entities(self) -> Dict[str, ExtractedEntity]:
        """Get all cached entities.

        Returns:
            Dict of entity_id -> ExtractedEntity
        """
        if not self.cache_enabled:
            return {}

        await self._ensure_initialized()

        entities = {}
        async with aiosqlite.connect(self._db.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT entity_id, name, entity_type, aliases, description,
                       frequency, paper_ids
                FROM entities
                """
            )
            rows = await cursor.fetchall()

            for row in rows:
                entities[row["entity_id"]] = ExtractedEntity(
                    entity_id=row["entity_id"],
                    name=row["name"],
                    entity_type=row["entity_type"],
                    aliases=json.loads(row["aliases"]) if row["aliases"] else [],
                    description=row["description"] or "",
                    frequency=row["frequency"] or 1,
                    paper_ids=json.loads(row["paper_ids"]) if row["paper_ids"] else [],
                )

        return entities

    async def get_all_mentions(self) -> List[EntityMention]:
        """Get all cached mentions.

        Returns:
            List of EntityMention
        """
        if not self.cache_enabled:
            return []

        await self._ensure_initialized()

        mentions = []
        async with aiosqlite.connect(self._db.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT entity_id, paper_id, context, section, confidence
                FROM entity_mentions
                """
            )
            rows = await cursor.fetchall()

            for row in rows:
                mentions.append(
                    EntityMention(
                        entity_id=row["entity_id"],
                        paper_id=row["paper_id"],
                        context=row["context"] or "",
                        section=row["section"] or "",
                        confidence=row["confidence"] or 0.9,
                    )
                )

        return mentions

    # =========================================================================
    # Graph Edge Operations
    # =========================================================================

    async def save_edges(self, edges: List[GraphEdge]) -> None:
        """Save graph edges to cache.

        Args:
            edges: List of graph edges
        """
        if not self.cache_enabled:
            return

        await self._ensure_initialized()

        async with aiosqlite.connect(self._db.db_path) as db:
            for edge in edges:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO graph_edges
                        (source_id, target_id, edge_type, weight, paper_ids)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        edge["source_id"],
                        edge["target_id"],
                        edge["edge_type"],
                        edge.get("weight", 1.0),
                        json.dumps(edge.get("paper_ids", [])),
                    ),
                )
            await db.commit()

        logger.info(f"Saved {len(edges)} edges to cache")

    async def get_all_edges(self) -> List[GraphEdge]:
        """Get all cached edges.

        Returns:
            List of GraphEdge
        """
        if not self.cache_enabled:
            return []

        await self._ensure_initialized()

        edges = []
        async with aiosqlite.connect(self._db.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT source_id, target_id, edge_type, weight, paper_ids FROM graph_edges"
            )
            rows = await cursor.fetchall()

            for row in rows:
                edges.append(
                    GraphEdge(
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        edge_type=row["edge_type"],
                        weight=row["weight"] or 1.0,
                        paper_ids=json.loads(row["paper_ids"]) if row["paper_ids"] else [],
                    )
                )

        return edges

    # =========================================================================
    # Community Operations
    # =========================================================================

    async def save_communities(self, communities: List[Community]) -> None:
        """Save communities to cache.

        Args:
            communities: List of communities
        """
        if not self.cache_enabled:
            return

        await self._ensure_initialized()

        async with aiosqlite.connect(self._db.db_path) as db:
            # Clear existing communities (they're regenerated each run)
            await db.execute("DELETE FROM communities")

            for comm in communities:
                await db.execute(
                    """
                    INSERT INTO communities
                        (community_id, level, entity_ids, paper_ids, summary,
                         parent_id, children_ids, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        comm["community_id"],
                        comm["level"],
                        json.dumps(comm.get("entities", [])),
                        json.dumps(comm.get("papers", [])),
                        comm.get("summary", ""),
                        comm.get("parent_id"),
                        json.dumps(comm.get("children_ids", [])),
                        datetime.now().isoformat(),
                    ),
                )
            await db.commit()

        logger.info(f"Saved {len(communities)} communities to cache")

    async def get_all_communities(self) -> List[Community]:
        """Get all cached communities.

        Returns:
            List of Community
        """
        if not self.cache_enabled:
            return []

        await self._ensure_initialized()

        communities = []
        async with aiosqlite.connect(self._db.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT community_id, level, entity_ids, paper_ids, summary,
                       parent_id, children_ids
                FROM communities
                """
            )
            rows = await cursor.fetchall()

            for row in rows:
                communities.append(
                    Community(
                        community_id=row["community_id"],
                        level=row["level"],
                        entities=json.loads(row["entity_ids"]) if row["entity_ids"] else [],
                        papers=json.loads(row["paper_ids"]) if row["paper_ids"] else [],
                        summary=row["summary"] or "",
                        parent_id=row["parent_id"],
                        children_ids=json.loads(row["children_ids"]) if row["children_ids"] else [],
                    )
                )

        return communities

    # =========================================================================
    # Utility Operations
    # =========================================================================

    async def clear_all(self) -> None:
        """Clear all GraphRAG cache data."""
        if not self.cache_enabled:
            return

        await self._ensure_initialized()

        async with aiosqlite.connect(self._db.db_path) as db:
            await db.execute("DELETE FROM entity_mentions")
            await db.execute("DELETE FROM entities")
            await db.execute("DELETE FROM graph_edges")
            await db.execute("DELETE FROM communities")
            await db.commit()

        logger.info("Cleared all GraphRAG cache data")

    async def get_stats(self) -> Dict[str, Any]:
        """Get GraphRAG cache statistics.

        Returns:
            Dict with cache stats
        """
        if not self.cache_enabled:
            return {"cache_enabled": False}

        await self._ensure_initialized()

        stats = {"cache_enabled": True}
        async with aiosqlite.connect(self._db.db_path) as db:
            for table in ["entities", "entity_mentions", "graph_edges", "communities"]:
                cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                row = await cursor.fetchone()
                stats[f"{table}_count"] = row[0] if row else 0

            # Entity type distribution
            cursor = await db.execute(
                "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type"
            )
            rows = await cursor.fetchall()
            stats["entity_types"] = {row[0]: row[1] for row in rows}

        return stats


# Global cache instance
_graphrag_cache: Optional[GraphRAGCache] = None


def get_graphrag_cache(cache_enabled: bool = True) -> GraphRAGCache:
    """Get or create the global GraphRAG cache instance.

    Args:
        cache_enabled: Whether caching is enabled

    Returns:
        GraphRAGCache instance
    """
    global _graphrag_cache
    if _graphrag_cache is None:
        _graphrag_cache = GraphRAGCache(cache_enabled=cache_enabled)
    return _graphrag_cache


__all__ = ["GraphRAGCache", "get_graphrag_cache"]

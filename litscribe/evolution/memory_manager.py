"""MemoryManager — unified three-tier memory facade."""
from __future__ import annotations

from pathlib import Path

from litscribe.evolution.episodic import EpisodicMemory
from litscribe.evolution.procedural import ProceduralMemory
from litscribe.evolution.semantic import SemanticMemory
from litscribe.evolution.skill_evolver import SkillEvolver
from litscribe.store.sqlite import SQLiteStore
from litscribe.store.vectors import VectorStore


class MemoryManager:
    """Owns and wires together all three memory tiers plus the skill evolver."""

    def __init__(self, db_path: Path, chroma_path: Path, skills_dir: Path):
        self._sqlite = SQLiteStore(db_path)
        self._vectors = VectorStore(chroma_path)
        self.episodic = EpisodicMemory(self._sqlite)
        self.semantic = SemanticMemory(self._vectors)
        self.procedural = ProceduralMemory(skills_dir, self._vectors)
        self.evolver = SkillEvolver(self.episodic, self.procedural)

    async def initialize(self) -> None:
        """Initialize underlying stores (must be called before use)."""
        await self._sqlite.initialize()

    async def close(self) -> None:
        """Cleanly shut down underlying stores."""
        await self._sqlite.close()

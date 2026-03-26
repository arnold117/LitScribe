import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory for test data (SQLite, ChromaDB, skills)."""
    db_dir = tmp_path / "data"
    db_dir.mkdir()
    return db_dir


@pytest.fixture
def tmp_skills_dir(tmp_path):
    """Temporary directory for procedural skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    return skills_dir


@pytest.fixture(autouse=True)
def env_override(monkeypatch, tmp_data_dir):
    """Override environment for all tests."""
    monkeypatch.setenv("LITSCRIBE_DATA_DIR", str(tmp_data_dir))
    monkeypatch.setenv("LITSCRIBE_TESTING", "1")

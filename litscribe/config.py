from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    provider: str = "openai"  # openai | anthropic | ollama
    api_base: str = ""
    api_key: str = ""
    default_model: str = ""
    reasoning_model: str = ""
    task_models: dict[str, str] = field(default_factory=dict)


@dataclass
class ServiceConfig:
    ncbi_email: str = ""
    ncbi_api_key: str = ""
    semantic_scholar_api_key: str = ""
    zotero_api_key: str = ""
    zotero_library_id: str = ""
    unpaywall_email: str = ""
    max_concurrent_requests: int = 5


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    services: ServiceConfig = field(default_factory=ServiceConfig)
    data_dir: Path = field(default_factory=lambda: Path.home() / ".litscribe" / "data")
    skills_dir: Path = field(default_factory=lambda: Path.home() / ".litscribe" / "skills")
    graphrag_enabled: bool = True
    debug: bool = False

    def __init__(self, config_path: Path | None = None):
        self.llm = LLMConfig()
        self.services = ServiceConfig()
        self.data_dir = Path(os.getenv("LITSCRIBE_DATA_DIR", Path.home() / ".litscribe" / "data"))
        self.skills_dir = Path(os.getenv("LITSCRIBE_SKILLS_DIR", Path.home() / ".litscribe" / "skills"))
        self.graphrag_enabled = os.getenv("LITSCRIBE_GRAPHRAG_ENABLED", "1") == "1"
        self.debug = os.getenv("LITSCRIBE_DEBUG", "0") == "1"

        self._load_env()
        if config_path and config_path.exists():
            self._load_yaml(config_path)
        else:
            default_yaml = Path.home() / ".litscribe" / "config.yaml"
            if default_yaml.exists():
                self._load_yaml(default_yaml)

    def _load_env(self):
        self.llm.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.llm.api_key = (
            os.getenv("llm-key", "")
            or os.getenv("LLM_API_KEY", "")
        )
        self.llm.api_base = (
            os.getenv("llm-location", "")
            or os.getenv("LLM_API_BASE", "")
        )
        self.llm.default_model = (
            os.getenv("llm-model", "")
            or os.getenv("LLM_MODEL", "")
        )
        self.llm.reasoning_model = os.getenv(
            "LLM_REASONING_MODEL", self.llm.default_model
        )

        self.services.ncbi_email = os.getenv("NCBI_EMAIL", "")
        self.services.ncbi_api_key = os.getenv("NCBI_API_KEY", "")
        self.services.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.services.zotero_api_key = os.getenv("ZOTERO_API_KEY", "")
        self.services.zotero_library_id = os.getenv("ZOTERO_LIBRARY_ID", "")
        self.services.unpaywall_email = os.getenv("UNPAYWALL_EMAIL", "")

    def _load_yaml(self, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        llm_data = data.get("llm", {})
        if "provider" in llm_data:
            self.llm.provider = llm_data["provider"].lower()
        if "default_model" in llm_data:
            self.llm.default_model = llm_data["default_model"]
        if "api_base" in llm_data:
            self.llm.api_base = llm_data["api_base"]
        if "api_key" in llm_data:
            self.llm.api_key = llm_data["api_key"]
        if "reasoning_model" in llm_data:
            self.llm.reasoning_model = llm_data["reasoning_model"]
        if "task_models" in llm_data:
            self.llm.task_models.update(llm_data["task_models"])

        services_data = data.get("services", {})
        for key, val in services_data.items():
            if hasattr(self.services, key):
                setattr(self.services, key, val)

        if "graphrag_enabled" in data:
            self.graphrag_enabled = data["graphrag_enabled"]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "litscribe.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "vectors"

    def ensure_directories(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)

"""Configuration management for LitScribe."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration for LitScribe."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = Path(os.getenv("LITSCRIBE_DATA_DIR", PROJECT_ROOT / "data"))
    CACHE_DIR = Path(os.getenv("LITSCRIBE_CACHE_DIR", PROJECT_ROOT / "cache"))
    LOG_LEVEL = os.getenv("LITSCRIBE_LOG_LEVEL", "INFO")

    # LLM Configuration (via LiteLLM)
    LITELLM_MODEL = os.getenv("LITELLM_MODEL", "deepseek/deepseek-chat")
    LITELLM_REASONING_MODEL = os.getenv("LITELLM_REASONING_MODEL", "")  # e.g. "deepseek/deepseek-reasoner"
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")

    # Academic Search API Keys
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
    NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
    NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    # Zotero configuration
    ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY", "")
    ZOTERO_LIBRARY_ID = os.getenv("ZOTERO_LIBRARY_ID", "")
    ZOTERO_LIBRARY_TYPE = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    ZOTERO_DEFAULT_COLLECTION = os.getenv("ZOTERO_DEFAULT_COLLECTION", "")  # Sandbox collection
    ZOTERO_STORAGE_DIR = Path(os.getenv("ZOTERO_STORAGE_DIR", Path.home() / "Zotero" / "storage"))
    ZOTERO_LOCAL = os.getenv("ZOTERO_LOCAL", "false").lower() == "true"
    ZOTERO_LOCAL_PORT = int(os.getenv("ZOTERO_LOCAL_PORT", "23119"))
    ZOTERO_LOCAL_URL = f"http://localhost:{ZOTERO_LOCAL_PORT}"

    # Local LLM configuration
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen3:32b-4bit")
    MLX_ENABLED = os.getenv("MLX_ENABLED", "false").lower() == "true"
    MLX_MODEL_PATH = os.getenv("MLX_MODEL_PATH", "")

    # Cache settings
    CACHE_EXPIRATION_HOURS = int(os.getenv("CACHE_EXPIRATION_HOURS", "24"))

    # API rate limits
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    PUBMED_RATE_LIMIT = 10 if NCBI_API_KEY else 3  # requests per second
    SERPAPI_RATE_LIMIT = 1  # requests per second
    SEMANTIC_SCHOLAR_RATE_LIMIT = 1  # 1 request per second with API key

    # Development settings
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    TESTING = os.getenv("TESTING", "false").lower() == "true"

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        directories = [
            cls.DATA_DIR,
            cls.DATA_DIR / "pdfs",
            cls.DATA_DIR / "parsed",
            cls.DATA_DIR / "embeddings",
            cls.CACHE_DIR,
            cls.CACHE_DIR / "arxiv",
            cls.CACHE_DIR / "pubmed",
            cls.CACHE_DIR / "scholar",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_api_keys(cls) -> dict[str, bool]:
        """Validate that required API keys are present."""
        keys = {
            "deepseek": bool(cls.DEEPSEEK_API_KEY),
            "anthropic": bool(cls.ANTHROPIC_API_KEY),
            "serpapi": bool(cls.SERPAPI_KEY),
            "ncbi": bool(cls.NCBI_API_KEY),
            "zotero": bool(cls.ZOTERO_API_KEY),
            "semantic_scholar": bool(cls.SEMANTIC_SCHOLAR_API_KEY),
        }
        return keys

    @classmethod
    def get_summary(cls) -> dict[str, Any]:
        """Get configuration summary for debugging."""
        return {
            "project_root": str(cls.PROJECT_ROOT),
            "data_dir": str(cls.DATA_DIR),
            "cache_dir": str(cls.CACHE_DIR),
            "log_level": cls.LOG_LEVEL,
            "api_keys_configured": cls.validate_api_keys(),
            "local_llm": {
                "base_url": cls.LOCAL_LLM_BASE_URL,
                "model": cls.LOCAL_LLM_MODEL,
                "mlx_enabled": cls.MLX_ENABLED,
            },
            "zotero": {
                "local": cls.ZOTERO_LOCAL,
                "library_type": cls.ZOTERO_LIBRARY_TYPE,
            },
            "debug": cls.DEBUG,
            "testing": cls.TESTING,
        }


# Initialize directories on import
Config.ensure_directories()

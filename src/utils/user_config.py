"""User configuration persistence for LitScribe.

Saves and loads user preferences from ~/.litscribe/config.yaml.
CLI arguments override user config, which overrides system defaults.

Priority: CLI args > user config > system defaults
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default config directory
CONFIG_DIR = Path.home() / ".litscribe"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Default user preferences
DEFAULT_USER_CONFIG = {
    "sources": ["arxiv", "semantic_scholar", "pubmed"],
    "max_papers": 10,
    "review_type": "narrative",
    "citation_style": "APA",
    "language": "en",
    "batch_size": 20,
    "graphrag_enabled": True,
    "zotero": {
        "default_collection": "",
        "auto_save": False,
        "write_notes": False,
    },
    "export": {
        "format": "markdown",
        "style": "APA",
    },
}


def _ensure_config_dir() -> None:
    """Ensure ~/.litscribe directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_user_config() -> Dict[str, Any]:
    """Load user configuration from ~/.litscribe/config.yaml.

    Returns:
        User config dict, or defaults if file doesn't exist
    """
    if not CONFIG_FILE.exists():
        return dict(DEFAULT_USER_CONFIG)

    try:
        import yaml

        with open(CONFIG_FILE) as f:
            user_config = yaml.safe_load(f) or {}

        # Merge with defaults (user config overrides defaults)
        merged = dict(DEFAULT_USER_CONFIG)
        _deep_merge(merged, user_config)

        logger.info(f"Loaded user config from {CONFIG_FILE}")
        return merged

    except ImportError:
        logger.warning("PyYAML not installed, using default config")
        return dict(DEFAULT_USER_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to load user config: {e}")
        return dict(DEFAULT_USER_CONFIG)


def save_user_config(config: Dict[str, Any]) -> bool:
    """Save user configuration to ~/.litscribe/config.yaml.

    Args:
        config: Configuration dict to save

    Returns:
        True if saved successfully
    """
    try:
        import yaml

        _ensure_config_dir()

        with open(CONFIG_FILE, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Saved user config to {CONFIG_FILE}")
        return True

    except ImportError:
        logger.error("PyYAML not installed, cannot save config")
        return False
    except Exception as e:
        logger.error(f"Failed to save user config: {e}")
        return False


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific config value using dot notation.

    Args:
        key: Config key (e.g., "sources", "zotero.default_collection")
        default: Default value if key not found

    Returns:
        Config value
    """
    config = load_user_config()
    keys = key.split(".")
    value = config

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value


def set_config_value(key: str, value: Any) -> bool:
    """Set a specific config value using dot notation.

    Args:
        key: Config key (e.g., "sources", "zotero.default_collection")
        value: Value to set

    Returns:
        True if saved successfully
    """
    config = load_user_config()
    keys = key.split(".")
    target = config

    # Navigate to parent
    for k in keys[:-1]:
        if k not in target or not isinstance(target[k], dict):
            target[k] = {}
        target = target[k]

    target[keys[-1]] = value
    return save_user_config(config)


def merge_with_cli_args(
    cli_sources: Optional[List[str]] = None,
    cli_max_papers: Optional[int] = None,
    cli_review_type: Optional[str] = None,
    cli_batch_size: Optional[int] = None,
    cli_graphrag: Optional[bool] = None,
    cli_local_files: Optional[List[str]] = None,
    cli_language: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge user config with CLI arguments.

    CLI arguments take priority over user config.

    Args:
        cli_sources: Sources from CLI --sources
        cli_max_papers: Max papers from CLI --papers
        cli_review_type: Review type from CLI --type
        cli_batch_size: Batch size from CLI --batch-size
        cli_graphrag: GraphRAG enabled from CLI
        cli_local_files: Local files from CLI --local-files
        cli_language: Language from CLI --lang

    Returns:
        Merged configuration dict
    """
    config = load_user_config()

    # CLI args override user config
    if cli_sources is not None:
        config["sources"] = cli_sources
    if cli_max_papers is not None:
        config["max_papers"] = cli_max_papers
    if cli_review_type is not None:
        config["review_type"] = cli_review_type
    if cli_batch_size is not None:
        config["batch_size"] = cli_batch_size
    if cli_graphrag is not None:
        config["graphrag_enabled"] = cli_graphrag
    if cli_local_files is not None:
        config["local_files"] = cli_local_files
    if cli_language is not None:
        config["language"] = cli_language

    return config


def _deep_merge(base: dict, override: dict) -> None:
    """Deep merge override into base dict (in-place).

    Args:
        base: Base dict to merge into
        override: Override dict with priority values
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_config_summary() -> str:
    """Get a formatted summary of current config for display.

    Returns:
        Formatted config string
    """
    config = load_user_config()
    lines = [f"Config file: {CONFIG_FILE}"]
    lines.append(f"  sources: {', '.join(config.get('sources', []))}")
    lines.append(f"  max_papers: {config.get('max_papers', 10)}")
    lines.append(f"  review_type: {config.get('review_type', 'narrative')}")
    lines.append(f"  citation_style: {config.get('citation_style', 'APA')}")
    lines.append(f"  language: {config.get('language', 'en')}")
    lines.append(f"  batch_size: {config.get('batch_size', 20)}")
    lines.append(f"  graphrag_enabled: {config.get('graphrag_enabled', True)}")

    zotero = config.get("zotero", {})
    lines.append(f"  zotero.default_collection: {zotero.get('default_collection', '')}")
    lines.append(f"  zotero.auto_save: {zotero.get('auto_save', False)}")

    return "\n".join(lines)

"""Caching utilities for LitScribe."""

import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from .config import Config


class CacheManager:
    """Simple file-based cache manager."""

    def __init__(self, cache_subdir: str = ""):
        """
        Initialize cache manager.

        Args:
            cache_subdir: Subdirectory within cache (e.g., 'arxiv', 'pubmed')
        """
        self.cache_dir = Config.CACHE_DIR / cache_subdir if cache_subdir else Config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiration_hours = Config.CACHE_EXPIRATION_HOURS

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Use MD5 hash of key as filename to handle special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def _is_expired(self, cache_path: Path) -> bool:
        """Check if cache file is expired."""
        if not cache_path.exists():
            return True

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiration = mtime + timedelta(hours=self.expiration_hours)
        return datetime.now() > expiration

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)

        if self._is_expired(cache_path):
            if cache_path.exists():
                cache_path.unlink()  # Delete expired cache
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                return data
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be pickle-able)
        """
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Warning: Failed to cache {key}: {e}")

    def delete(self, key: str) -> None:
        """Delete a cache entry."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

    def clear(self) -> int:
        """
        Clear all cache in this subdirectory.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
            count += 1
        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired cache files.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            if self._is_expired(cache_file):
                cache_file.unlink()
                count += 1
        return count

    def get_stats(self) -> dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_files = len(cache_files)
        total_size = sum(f.stat().st_size for f in cache_files)
        expired_files = sum(1 for f in cache_files if self._is_expired(f))

        return {
            "total_files": total_files,
            "expired_files": expired_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }

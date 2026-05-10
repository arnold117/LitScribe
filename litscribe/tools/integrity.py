from __future__ import annotations

import hashlib
import hmac
import logging
import os

logger = logging.getLogger(__name__)

_SECRET_KEY: bytes | None = None


def _get_secret() -> bytes:
    global _SECRET_KEY
    if _SECRET_KEY is None:
        key_str = os.getenv("LITSCRIBE_INTEGRITY_KEY", "")
        if not key_str:
            key_path = os.path.expanduser("~/.litscribe/.integrity_key")
            try:
                with open(key_path, "rb") as f:
                    _SECRET_KEY = f.read()
            except FileNotFoundError:
                _SECRET_KEY = os.urandom(32)
                os.makedirs(os.path.dirname(key_path), exist_ok=True)
                with open(key_path, "wb") as f:
                    f.write(_SECRET_KEY)
                os.chmod(key_path, 0o600)
                logger.info(f"Generated integrity key at {key_path}")
        else:
            _SECRET_KEY = key_str.encode()
    return _SECRET_KEY


def sign_record(content: str) -> str:
    secret = _get_secret()
    return hmac.new(secret, content.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def verify_record(content: str, signature: str) -> bool:
    expected = sign_record(content)
    return hmac.compare_digest(expected, signature)


def sign_finding(domain: str, topic: str, finding: str) -> str:
    return sign_record(f"{domain}:{topic}:{finding}")


def verify_finding(domain: str, topic: str, finding: str, signature: str) -> bool:
    return verify_record(f"{domain}:{topic}:{finding}", signature)

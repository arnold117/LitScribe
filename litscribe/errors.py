"""Error handling and retry utilities for LitScribe agents.

This module provides a structured approach to error handling with:
- Typed error classifications
- Retry decorators with exponential backoff
- Graceful degradation strategies
"""

import asyncio
import logging
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorType(Enum):
    """Classification of errors for appropriate handling."""

    # Transient errors - should retry
    API_RATE_LIMIT = "rate_limit"
    API_TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"

    # Recoverable errors - may need fallback
    PDF_NOT_FOUND = "pdf_not_found"
    PDF_PARSE_ERROR = "parse_error"
    LLM_ERROR = "llm_error"

    # Non-recoverable errors
    INVALID_INPUT = "invalid_input"
    CONFIGURATION_ERROR = "config_error"
    UNKNOWN_ERROR = "unknown_error"


class AgentError(Exception):
    """Base exception for agent-related errors."""

    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        recoverable: bool = True,
        original_error: Optional[Exception] = None,
        context: Optional[dict] = None,
    ):
        self.error_type = error_type
        self.message = message
        self.recoverable = recoverable
        self.original_error = original_error
        self.context = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        return f"[{self.error_type.value}] {self.message}"

    def to_dict(self) -> dict:
        """Convert to dictionary for state storage."""
        return {
            "type": self.error_type.value,
            "message": self.message,
            "recoverable": self.recoverable,
            "context": self.context,
        }


class RateLimitError(AgentError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(ErrorType.API_RATE_LIMIT, message, recoverable=True, **kwargs)


class TimeoutError(AgentError):
    """Raised when API request times out."""

    def __init__(self, message: str = "Request timed out", **kwargs):
        super().__init__(ErrorType.API_TIMEOUT, message, recoverable=True, **kwargs)


class PDFNotFoundError(AgentError):
    """Raised when PDF cannot be found or accessed."""

    def __init__(self, message: str = "PDF not found", **kwargs):
        super().__init__(ErrorType.PDF_NOT_FOUND, message, recoverable=True, **kwargs)


class PDFParseError(AgentError):
    """Raised when PDF parsing fails."""

    def __init__(self, message: str = "PDF parsing failed", **kwargs):
        super().__init__(ErrorType.PDF_PARSE_ERROR, message, recoverable=True, **kwargs)


class LLMError(AgentError):
    """Raised when LLM call fails."""

    def __init__(self, message: str = "LLM call failed", **kwargs):
        super().__init__(ErrorType.LLM_ERROR, message, recoverable=True, **kwargs)


# Retry decorators with different strategies

def retry_on_rate_limit(
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 60.0,
):
    """Retry decorator for rate-limited API calls.

    Uses exponential backoff optimized for rate limit scenarios.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((RateLimitError, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Rate limited, retrying in {retry_state.next_action.sleep:.1f}s "
            f"(attempt {retry_state.attempt_number}/{max_attempts})"
        ),
    )


def retry_on_transient_error(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
):
    """Retry decorator for transient errors (network issues, timeouts)."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((RateLimitError, TimeoutError, ConnectionError)),
    )


def retry_llm_call(
    max_attempts: int = 2,
    min_wait: float = 5.0,
    max_wait: float = 120.0,
):
    """Retry decorator for LLM calls with longer backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((LLMError, RateLimitError)),
    )


async def with_fallback(
    primary_fn: Callable[..., T],
    fallback_fn: Callable[..., T],
    *args,
    error_types: tuple = (AgentError,),
    **kwargs,
) -> T:
    """Execute primary function with fallback on failure.

    Args:
        primary_fn: Primary async function to try
        fallback_fn: Fallback async function if primary fails
        *args: Arguments to pass to both functions
        error_types: Tuple of exception types to catch
        **kwargs: Keyword arguments to pass to both functions

    Returns:
        Result from primary or fallback function

    Example:
        result = await with_fallback(
            parse_pdf_with_marker,
            parse_pdf_with_pymupdf,
            pdf_path,
            error_types=(PDFParseError,)
        )
    """
    try:
        return await primary_fn(*args, **kwargs)
    except error_types as e:
        logger.warning(f"Primary function failed: {e}, trying fallback")
        return await fallback_fn(*args, **kwargs)


def safe_execute(
    default_value: Any = None,
    error_types: tuple = (Exception,),
    log_error: bool = True,
):
    """Decorator for safe execution with default value on failure.

    Args:
        default_value: Value to return on failure
        error_types: Exception types to catch
        log_error: Whether to log the error
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except error_types as e:
                if log_error:
                    logger.error(f"{func.__name__} failed: {e}")
                return default_value
        return wrapper
    return decorator


class ErrorCollector:
    """Collect and manage errors during workflow execution."""

    def __init__(self):
        self.errors: list[AgentError] = []

    def add(self, error: AgentError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)
        logger.error(f"Error collected: {error}")

    def add_from_exception(
        self,
        exception: Exception,
        error_type: ErrorType = ErrorType.UNKNOWN_ERROR,
        context: Optional[dict] = None,
    ) -> None:
        """Create and add an AgentError from a generic exception."""
        error = AgentError(
            error_type=error_type,
            message=str(exception),
            original_error=exception,
            context=context,
        )
        self.add(error)

    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0

    def has_unrecoverable_errors(self) -> bool:
        """Check if any non-recoverable errors exist."""
        return any(not e.recoverable for e in self.errors)

    def get_error_messages(self) -> list[str]:
        """Get list of error messages for state storage."""
        return [str(e) for e in self.errors]

    def clear(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()


# Rate limiter for API calls
class AsyncRateLimiter:
    """Simple async rate limiter for API calls."""

    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until we can make another call."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_call
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            self.last_call = asyncio.get_event_loop().time()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        pass


# Pre-configured rate limiters for different APIs
semantic_scholar_limiter = AsyncRateLimiter(calls_per_second=1.0)  # 1 req/sec with API key
arxiv_limiter = AsyncRateLimiter(calls_per_second=3.0)  # 3 req/sec
pubmed_limiter = AsyncRateLimiter(calls_per_second=10.0)  # 10 req/sec with API key

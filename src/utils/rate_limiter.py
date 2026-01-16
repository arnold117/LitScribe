"""Rate limiting utilities."""

import asyncio
import time
from collections import deque
from typing import Callable, TypeVar

T = TypeVar("T")


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_second: float):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0

    def wait_if_needed(self) -> None:
        """Block if necessary to maintain rate limit."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    async def wait_if_needed_async(self) -> None:
        """Async version of wait_if_needed."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = time.time()

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to rate limit a function."""

        def wrapper(*args, **kwargs) -> T:
            self.wait_if_needed()
            return func(*args, **kwargs)

        return wrapper


class SlidingWindowRateLimiter:
    """Rate limiter using sliding window algorithm (more accurate)."""

    def __init__(self, requests_per_second: float, window_size: int = 10):
        """
        Initialize sliding window rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            window_size: Number of recent requests to track
        """
        self.requests_per_second = requests_per_second
        self.window_size = window_size
        self.requests: deque[float] = deque(maxlen=window_size)

    def wait_if_needed(self) -> None:
        """Block if necessary to maintain rate limit."""
        current_time = time.time()

        # Remove requests outside the 1-second window
        cutoff_time = current_time - 1.0
        while self.requests and self.requests[0] < cutoff_time:
            self.requests.popleft()

        # Check if we've hit the limit
        if len(self.requests) >= self.requests_per_second:
            # Wait until the oldest request is 1 second old
            sleep_time = (self.requests[0] + 1.0) - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.requests.append(time.time())

    async def wait_if_needed_async(self) -> None:
        """Async version of wait_if_needed."""
        current_time = time.time()

        # Remove requests outside the 1-second window
        cutoff_time = current_time - 1.0
        while self.requests and self.requests[0] < cutoff_time:
            self.requests.popleft()

        # Check if we've hit the limit
        if len(self.requests) >= self.requests_per_second:
            # Wait until the oldest request is 1 second old
            sleep_time = (self.requests[0] + 1.0) - current_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.requests.append(time.time())

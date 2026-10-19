"""A circuit breaker for a provider that keeps failing."""

import math
import time
from collections.abc import Callable
from typing import Literal

BreakerState = Literal["closed", "open", "half_open"]


class BreakerOpenError(RuntimeError):
    """The breaker is open: fail fast instead of calling the provider."""


class CircuitBreaker:
    """Open after N consecutive failures; fail fast until a cooldown, then probe once.

    The breaker is synchronous and single-threaded (one event loop): ``allow`` and the
    ``record_*`` calls contain no ``await``, so they cannot interleave.
    """

    def __init__(
        self,
        *,
        failures: int,
        cooldown: float,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if failures < 1:
            raise ValueError("the breaker needs at least one failure")
        if cooldown <= 0:
            raise ValueError("the breaker cooldown must be positive")
        self._failure_limit = failures
        self._cooldown = cooldown
        self._now = now
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> BreakerState:
        """The state for reporting; ``half_open`` does not mean ``allow`` will pass."""
        if self._opened_at is None:
            return "closed"
        if self._probing:
            return "half_open"
        if self._now() < self._opened_at + self._cooldown:
            return "open"
        return "half_open"

    def retry_after(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self._cooldown - (self._now() - self._opened_at))

    def allow(self) -> None:
        """Let a call through, or fail fast; after the cooldown, let one probe through."""
        if self._opened_at is None:
            return
        if self._probing:
            raise BreakerOpenError("a recovery probe is already in flight")
        if self._now() < self._opened_at + self._cooldown:
            raise BreakerOpenError(f"retry in {math.ceil(self.retry_after())} s")
        self._probing = True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probing = False

    def abort_probe(self) -> None:
        """A probe ended without a provider verdict: reopen with a fresh cooldown."""
        if not self._probing:
            return
        self._probing = False
        self._opened_at = self._now()

    def record_failure(self) -> None:
        if self._probing:
            self._probing = False
            self._opened_at = self._now()
            return
        if self._opened_at is not None:
            return
        self._failures += 1
        if self._failures >= self._failure_limit:
            self._opened_at = self._now()

import pytest

from open_endurance_coach.clients.breaker import BreakerOpenError, CircuitBreaker


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def open_breaker(*, failures: int = 2, cooldown: float = 10.0) -> tuple[CircuitBreaker, Clock]:
    clock = Clock()
    breaker = CircuitBreaker(failures=failures, cooldown=cooldown, now=clock)
    for _ in range(failures):
        breaker.allow()
        breaker.record_failure()
    return breaker, clock


def test_the_breaker_opens_after_the_configured_failures() -> None:
    breaker, _ = open_breaker()

    assert breaker.state == "open"
    with pytest.raises(BreakerOpenError, match="retry in"):
        breaker.allow()


def test_it_fails_fast_while_open() -> None:
    breaker, clock = open_breaker(cooldown=10.0)
    clock.now += 9.0

    assert breaker.state == "open"
    with pytest.raises(BreakerOpenError):
        breaker.allow()


def test_the_probe_closes_the_breaker_after_the_cooldown() -> None:
    breaker, clock = open_breaker(cooldown=10.0)
    clock.now += 10.0

    assert breaker.state == "half_open"
    breaker.allow()
    with pytest.raises(BreakerOpenError, match="probe"):
        breaker.allow()
    breaker.record_success()

    assert breaker.state == "closed"


def test_a_failed_probe_reopens_the_breaker() -> None:
    breaker, clock = open_breaker(cooldown=10.0)
    clock.now += 10.0
    breaker.allow()

    breaker.record_failure()

    assert breaker.state == "open"
    with pytest.raises(BreakerOpenError):
        breaker.allow()
    clock.now += 10.0
    assert breaker.state == "half_open"


def test_a_success_resets_the_failure_count() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failures=2, cooldown=10.0, now=clock)
    breaker.allow()
    breaker.record_failure()
    breaker.record_success()
    breaker.allow()
    breaker.record_failure()

    assert breaker.state == "closed"


def test_the_breaker_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        CircuitBreaker(failures=0, cooldown=10.0)
    with pytest.raises(ValueError):
        CircuitBreaker(failures=1, cooldown=0.0)


def test_an_aborted_probe_reopens_the_breaker() -> None:
    breaker, clock = open_breaker(cooldown=10.0)
    clock.now += 10.0
    breaker.allow()

    breaker.abort_probe()

    assert breaker.state == "open"
    clock.now += 10.0
    assert breaker.state == "half_open"


def test_the_retry_message_never_reads_zero_seconds() -> None:
    breaker, clock = open_breaker(cooldown=10.0)
    clock.now += 9.7

    with pytest.raises(BreakerOpenError, match="retry in 1 s"):
        breaker.allow()
    assert breaker.retry_after() == pytest.approx(0.3)

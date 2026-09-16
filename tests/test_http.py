from datetime import UTC, datetime

import httpx

from open_endurance_coach.clients.http import parse_retry_after

FUTURE = "Wed, 21 Oct 2026 07:28:00 GMT"


def test_numeric_retry_after_is_used_as_seconds() -> None:
    headers = httpx.Headers({"Retry-After": "5"})
    assert parse_retry_after(headers, default=1.0) == 5.0


def test_http_date_retry_after_is_a_delay_from_now() -> None:
    headers = httpx.Headers({"Retry-After": FUTURE})
    now = datetime(2026, 10, 21, 7, 27, 0, tzinfo=UTC)
    assert parse_retry_after(headers, default=1.0, now=now) == 60.0


def test_past_http_date_retry_after_clamps_to_zero() -> None:
    headers = httpx.Headers({"Retry-After": FUTURE})
    now = datetime(2026, 10, 22, 7, 28, 0, tzinfo=UTC)
    assert parse_retry_after(headers, default=1.0, now=now) == 0.0


def test_unknown_retry_after_uses_the_default() -> None:
    headers = httpx.Headers({"Retry-After": "not-a-date"})
    assert parse_retry_after(headers, default=2.5) == 2.5

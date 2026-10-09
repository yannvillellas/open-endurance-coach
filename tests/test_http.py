from datetime import UTC, datetime

import httpx

from open_endurance_coach.clients.http import (
    MAX_RETRY_AFTER_SECONDS,
    error_detail,
    parse_retry_after,
)

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


def test_huge_retry_after_is_capped() -> None:
    headers = httpx.Headers({"Retry-After": "999999999"})
    assert parse_retry_after(headers, default=1.0) == MAX_RETRY_AFTER_SECONDS


def test_non_finite_retry_after_uses_the_default() -> None:
    assert parse_retry_after(httpx.Headers({"Retry-After": "inf"}), default=2.5) == 2.5
    assert parse_retry_after(httpx.Headers({"Retry-After": "nan"}), default=2.5) == 2.5


def test_far_future_http_date_is_capped() -> None:
    headers = httpx.Headers({"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"})
    now = datetime(2026, 10, 21, 7, 27, 0, tzinfo=UTC)
    assert parse_retry_after(headers, default=1.0, now=now) == MAX_RETRY_AFTER_SECONDS


def test_error_detail_is_a_single_line() -> None:
    response = httpx.Response(500, json={"message": "boom\nforged line"})
    assert error_detail(response) == "boom forged line"

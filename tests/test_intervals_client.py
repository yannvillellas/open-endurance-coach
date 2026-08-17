import base64

import httpx
import pytest

from open_endurance_coach.clients.intervals import (
    BROWSER_USER_AGENT,
    IntervalsApiError,
    IntervalsClient,
    RateLimits,
)
from open_endurance_coach.config import Settings

from .fakes import RecordingSleep


def make_client(
    settings: Settings,
    responses: list[httpx.Response],
    sleep: RecordingSleep | None = None,
) -> tuple[IntervalsClient, list[httpx.Request]]:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        assert responses, f"unexpected request: {request.method} {request.url}"
        return responses.pop(0)

    client = IntervalsClient(
        settings,
        transport=httpx.MockTransport(handler),
        sleep=sleep or RecordingSleep(),
    )
    return client, captured


async def test_basic_auth_and_browser_user_agent(settings: Settings) -> None:
    client, captured = make_client(settings, [httpx.Response(200, json=[])])
    await client.list_activities("2026-08-01", "2026-08-17")
    request = captured[0]
    expected_auth = "Basic " + base64.b64encode(b"API_KEY:test-intervals-key").decode()
    assert request.headers["authorization"] == expected_auth
    assert request.headers["user-agent"] == BROWSER_USER_AGENT
    await client.aclose()


async def test_list_activities_url_and_params(settings: Settings) -> None:
    client, captured = make_client(settings, [httpx.Response(200, json=[{"id": "i1"}])])
    result = await client.list_activities("2026-08-01", "2026-08-17", fields=["id", "type"])
    request = captured[0]
    assert str(request.url).startswith("https://intervals.icu/api/v1/athlete/12345/activities")
    assert request.url.params["oldest"] == "2026-08-01"
    assert request.url.params["newest"] == "2026-08-17"
    assert request.url.params["fields"] == "id,type"
    assert result == [{"id": "i1"}]
    await client.aclose()


async def test_rate_limit_headers_parsed(settings: Settings) -> None:
    client, _ = make_client(
        settings,
        [
            httpx.Response(
                200,
                json=[],
                headers={
                    "X-RateLimit-Limit": "2500,5000",
                    "X-RateLimit-Remaining": "2499,4990",
                },
            )
        ],
    )
    await client.list_activities("2026-08-01", "2026-08-17")
    limits = client.rate_limits
    assert limits.limit_15m == 2500
    assert limits.remaining_15m == 2499
    assert limits.limit_daily == 5000
    assert limits.remaining_daily == 4990
    await client.aclose()


async def test_429_retry_then_success(settings: Settings) -> None:
    sleep = RecordingSleep()
    client, captured = make_client(
        settings,
        [
            httpx.Response(429, headers={"Retry-After": "1"}, json={}),
            httpx.Response(200, json=[{"id": "i1"}]),
        ],
        sleep=sleep,
    )
    result = await client.list_activities("2026-08-01", "2026-08-17")
    assert len(captured) == 2
    assert sleep.calls == [1.0]
    assert result == [{"id": "i1"}]
    await client.aclose()


async def test_5xx_retry_then_success(settings: Settings) -> None:
    sleep = RecordingSleep()
    client, captured = make_client(
        settings,
        [httpx.Response(500), httpx.Response(200, json={})],
        sleep=sleep,
    )
    await client.get_athlete()
    assert len(captured) == 2
    assert sleep.calls == [0.0]
    await client.aclose()


async def test_4xx_raises(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(403, json={"message": "denied"})])
    with pytest.raises(IntervalsApiError) as excinfo:
        await client.list_activities("2026-08-01", "2026-08-17")
    assert excinfo.value.status_code == 403
    await client.aclose()


async def test_create_event_posts_payload(settings: Settings) -> None:
    payload = {"category": "WORKOUT", "start_date_local": "2026-08-20T00:00:00", "name": "X"}
    client, captured = make_client(settings, [httpx.Response(200, json={"id": 9, **payload})])
    result = await client.create_event(payload)
    request = captured[0]
    assert request.method == "POST"
    assert str(request.url).endswith("/athlete/12345/events")
    assert request.content and payload["name"] in request.content.decode()
    assert result["id"] == 9
    await client.aclose()


async def test_update_and_delete_event_paths(settings: Settings) -> None:
    client, captured = make_client(
        settings,
        [httpx.Response(200, json={"id": 9}), httpx.Response(204)],
    )
    await client.update_event("9", {"name": "Y"})
    await client.delete_event("9")
    assert captured[0].method == "PUT"
    assert str(captured[0].url).endswith("/athlete/12345/events/9")
    assert captured[1].method == "DELETE"
    assert str(captured[1].url).endswith("/athlete/12345/events/9")
    await client.aclose()


async def test_get_activity_with_intervals(settings: Settings) -> None:
    client, captured = make_client(settings, [httpx.Response(200, json={"id": "i5"})])
    await client.get_activity("i5")
    request = captured[0]
    assert str(request.url).startswith("https://intervals.icu/api/v1/activity/i5")
    assert request.url.params["intervals"] == "true"
    await client.aclose()


def test_rate_limits_from_headers_partial() -> None:
    limits = RateLimits.from_headers(httpx.Headers({"X-RateLimit-Limit": "2500,5000"}))
    assert limits.limit_15m == 2500
    assert limits.remaining_15m is None

import asyncio
import base64
from typing import Any

import httpx
import pytest

from open_endurance_coach.clients.intervals import (
    BROWSER_USER_AGENT,
    RATE_LIMIT_COOLDOWN_SECONDS,
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
    clock: Any = None,
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
        clock=clock,
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
    result = await client.list_activities("2026-08-01", "2026-08-17")
    request = captured[0]
    assert str(request.url).startswith("https://intervals.icu/api/v1/athlete/12345/activities")
    assert request.url.params["oldest"] == "2026-08-01"
    assert request.url.params["newest"] == "2026-08-17"
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


async def test_429_http_date_retry_after_clamps_a_past_date(settings: Settings) -> None:
    sleep = RecordingSleep()
    client, captured = make_client(
        settings,
        [
            httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2020 07:28:00 GMT"}, json={}),
            httpx.Response(200, json=[{"id": "i1"}]),
        ],
        sleep=sleep,
    )
    result = await client.list_activities("2026-08-01", "2026-08-17")
    assert len(captured) == 2
    assert sleep.calls == [0.0]
    assert result == [{"id": "i1"}]
    await client.aclose()


async def test_429_exhaustion_with_http_date_raises_cleanly(settings: Settings) -> None:
    sleep = RecordingSleep()
    client, captured = make_client(
        settings,
        [httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2020 07:28:00 GMT"}, json={})]
        * 4,
        sleep=sleep,
    )
    with pytest.raises(IntervalsApiError) as excinfo:
        await client.list_activities("2026-08-01", "2026-08-17")
    assert excinfo.value.status_code == 429
    assert len(captured) == 4
    assert sleep.calls == [0.0, 0.0, 0.0]
    await client.aclose()


async def test_5xx_retry_then_success(settings: Settings) -> None:
    sleep = RecordingSleep()
    client, captured = make_client(
        settings,
        [httpx.Response(500), httpx.Response(200, json=[])],
        sleep=sleep,
    )
    await client.list_activities("2026-08-01", "2026-08-17")
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


async def test_activity_ids_cannot_escape_the_request_path(settings: Settings) -> None:
    client, captured = make_client(settings, [httpx.Response(200, json={"id": "i5"})])
    await client.get_activity("../../athlete/i999/events")
    raw = captured[0].url.raw_path.split(b"?")[0]
    assert raw == b"/api/v1/activity/..%2F..%2Fathlete%2Fi999%2Fevents"
    await client.aclose()


async def test_event_ids_cannot_escape_the_request_path(settings: Settings) -> None:
    client, captured = make_client(settings, [httpx.Response(200, json={})])
    await client.get_event("../../athlete/i999/events")
    raw = captured[0].url.raw_path.split(b"?")[0]
    assert raw == b"/api/v1/athlete/12345/events/..%2F..%2Fathlete%2Fi999%2Fevents"
    await client.aclose()


async def test_get_activity_streams_path_params_and_parsing(settings: Settings) -> None:
    payload = [
        {"type": "time", "data": [0, 1, 2]},
        {"type": "heartrate", "data": [90, 95, 100]},
        {"type": "distance", "data": []},
    ]
    client, captured = make_client(settings, [httpx.Response(200, json=payload)])
    streams = await client.get_activity_streams("i7", ["time", "heartrate", "distance"])
    request = captured[0]
    assert str(request.url).startswith("https://intervals.icu/api/v1/activity/i7/streams")
    assert request.url.params["types"] == "time,heartrate,distance"
    assert streams == {"time": [0, 1, 2], "heartrate": [90, 95, 100], "distance": []}
    await client.aclose()


async def test_get_activity_streams_skips_non_list_data(settings: Settings) -> None:
    payload = [
        {"type": "time", "data": [0, 1, 2]},
        {"type": "heartrate", "data": 5},
        {"type": "distance", "data": None},
    ]
    client, _ = make_client(settings, [httpx.Response(200, json=payload)])
    streams = await client.get_activity_streams("i7", ["time", "heartrate", "distance"])
    assert streams == {"time": [0, 1, 2]}
    await client.aclose()


async def test_get_activity_streams_rejects_a_non_json_body(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, text="<html>oops</html>")])
    with pytest.raises(IntervalsApiError, match="streams"):
        await client.get_activity_streams("i7", ["time"])
    await client.aclose()


async def test_get_activity_streams_rejects_a_non_list_body(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, json={"error": "boom"})])
    with pytest.raises(IntervalsApiError, match="streams"):
        await client.get_activity_streams("i7", ["time"])
    await client.aclose()


async def test_get_activity_streams_skips_malformed_rows(settings: Settings) -> None:
    payload = [{"type": "time", "data": [0, 1, 2]}, {"data": [9]}, "not-a-row"]
    client, _ = make_client(settings, [httpx.Response(200, json=payload)])
    streams = await client.get_activity_streams("i7", ["time"])
    assert streams == {"time": [0, 1, 2]}
    await client.aclose()


async def test_get_activity_streams_with_no_types(settings: Settings) -> None:
    client, captured = make_client(settings, [httpx.Response(200, json=[])])
    streams = await client.get_activity_streams("i7", [])
    assert streams == {}
    assert captured[0].url.params["types"] == ""
    await client.aclose()


async def test_get_activity_streams_keeps_the_last_row_per_type(settings: Settings) -> None:
    payload = [{"type": "time", "data": [0]}, {"type": "time", "data": [1]}]
    client, _ = make_client(settings, [httpx.Response(200, json=payload)])
    streams = await client.get_activity_streams("i7", ["time"])
    assert streams == {"time": [1]}
    await client.aclose()


def test_rate_limits_from_headers_partial() -> None:
    limits = RateLimits.from_headers(httpx.Headers({"X-RateLimit-Limit": "2500,5000"}))
    assert limits.limit_15m == 2500
    assert limits.remaining_15m is None


async def test_get_activity_rejects_a_non_dict_body(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, json=[1, 2])])
    with pytest.raises(IntervalsApiError, match="activity"):
        await client.get_activity("i5")
    await client.aclose()


async def test_object_endpoints_reject_a_non_json_body(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, text="<html>oops</html>")])
    with pytest.raises(IntervalsApiError, match="event"):
        await client.get_event("e5")
    await client.aclose()


async def test_create_event_rejects_a_non_dict_body(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, json="ok")])
    with pytest.raises(IntervalsApiError, match="event"):
        await client.create_event({"name": "x"})
    await client.aclose()


async def test_list_activities_rejects_a_non_json_body(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, text="<html>oops</html>")])
    with pytest.raises(IntervalsApiError, match="activities"):
        await client.list_activities("2024-01-01", "2024-02-01")
    await client.aclose()


async def test_update_event_rejects_a_non_dict_body(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, json="ok")])
    with pytest.raises(IntervalsApiError, match="event"):
        await client.update_event("e5", {"name": "x"})
    await client.aclose()


async def test_list_wellness_rejects_a_non_list_body(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, json={"error": "boom"})])
    with pytest.raises(IntervalsApiError, match="wellness"):
        await client.list_wellness("2024-01-01", "2024-02-01")
    await client.aclose()


async def test_sport_settings_rejects_a_non_list_body(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, json={"error": "boom"})])
    with pytest.raises(IntervalsApiError, match="sport-settings"):
        await client.get_sport_settings()
    await client.aclose()


async def test_get_athlete_summary_returns_list(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, json=[{"month": "x"}, {"month": "y"}])])
    result = await client.get_athlete_summary()
    assert isinstance(result, list)
    assert len(result) == 2
    await client.aclose()


async def test_429_exhaustion_sleeps_only_between_attempts(settings: Settings) -> None:
    sleep = RecordingSleep()
    client, captured = make_client(
        settings,
        [
            httpx.Response(429, headers={"Retry-After": "1"}, json={}),
            httpx.Response(429, headers={"Retry-After": "1"}, json={}),
            httpx.Response(429, headers={"Retry-After": "1"}, json={}),
            httpx.Response(429, headers={"Retry-After": "1"}, json={}),
        ],
        sleep=sleep,
    )
    with pytest.raises(IntervalsApiError) as excinfo:
        await client.list_activities("2026-08-01", "2026-08-17")
    assert excinfo.value.status_code == 429
    assert len(captured) == 4
    assert sleep.calls == [1.0, 1.0, 1.0]
    await client.aclose()


async def test_transport_error_retried_then_success(settings: Settings) -> None:
    sleep = RecordingSleep()
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if len(captured) == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json=[{"id": "i1"}])

    client = IntervalsClient(settings, transport=httpx.MockTransport(handler), sleep=sleep)
    result = await client.list_activities("2026-08-01", "2026-08-17")
    assert len(captured) == 2
    assert sleep.calls == [0.0]
    assert result == [{"id": "i1"}]
    await client.aclose()


async def test_redirects_are_followed(settings: Settings) -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if len(captured) == 1:
            return httpx.Response(302, headers={"Location": "https://intervals.icu/api/v1/moved"})
        return httpx.Response(200, json=[{"id": "i1"}])

    client = IntervalsClient(settings, transport=httpx.MockTransport(handler))
    result = await client.list_activities("2026-08-01", "2026-08-17")
    assert len(captured) == 2
    assert result == [{"id": "i1"}]
    await client.aclose()


async def test_get_athlete_summary_sends_window(settings: Settings) -> None:
    client, captured = make_client(
        settings, [httpx.Response(200, json=[{"date": "2024-01-30"}, {"date": "2024-01-23"}])]
    )
    result = await client.get_athlete_summary(start="2024-01-01", end="2024-01-31")
    assert isinstance(result, list)
    assert len(result) == 2
    assert captured[0].url.params["start"] == "2024-01-01"
    assert captured[0].url.params["end"] == "2024-01-31"
    await client.aclose()


async def test_get_athlete_summary_omits_absent_window(settings: Settings) -> None:
    client, captured = make_client(settings, [httpx.Response(200, json=[])])
    await client.get_athlete_summary()
    assert "start" not in captured[0].url.params
    await client.aclose()


async def test_get_athlete_summary_rejects_non_list_payload(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, json={"unexpected": True})])
    with pytest.raises(IntervalsApiError, match="unexpected athlete-summary payload"):
        await client.get_athlete_summary()
    await client.aclose()


async def test_list_activities_rejects_non_list_payload(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, json={"unexpected": True})])
    with pytest.raises(IntervalsApiError, match="unexpected activities payload"):
        await client.list_activities("2024-01-01", "2024-02-01")
    await client.aclose()


async def test_list_events_rejects_non_list_payload(settings: Settings) -> None:
    client, _ = make_client(settings, [httpx.Response(200, json={"unexpected": True})])
    with pytest.raises(IntervalsApiError, match="unexpected events payload"):
        await client.list_events("2024-01-01", "2024-02-01")
    await client.aclose()


async def test_error_messages_do_not_echo_the_response_body(settings: Settings) -> None:
    body = {"message": "denied", "athlete": {"name": "Private Athlete", "hr": 145}}
    client, _ = make_client(settings, [httpx.Response(403, json=body)])
    with pytest.raises(IntervalsApiError) as excinfo:
        await client.list_activities("2026-08-01", "2026-08-17")
    message = str(excinfo.value)
    assert "403" in message
    assert "denied" in message
    assert "Private Athlete" not in message
    await client.aclose()


async def test_concurrent_throttles_are_spaced(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings.model_copy(update={"requests_per_second": 2.0})
    clock = [0.0]
    sleep_starts: list[float] = []

    async def advancing_sleep(seconds: float) -> None:
        sleep_starts.append(clock[0])
        clock[0] += seconds

    monkeypatch.setattr("open_endurance_coach.clients.intervals.time.monotonic", lambda: clock[0])
    client = IntervalsClient(settings, sleep=advancing_sleep)
    client._last_request_at = 0.0

    await asyncio.gather(client._throttle(), client._throttle())

    assert sleep_starts == [0.0, 0.5]
    await client.aclose()


async def test_concurrent_throttles_do_not_hold_the_lock_while_sleeping(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings.model_copy(update={"requests_per_second": 2.0})
    monkeypatch.setattr("open_endurance_coach.clients.intervals.time.monotonic", lambda: 0.0)
    active = 0
    max_active = 0

    async def concurrent_sleep(seconds: float) -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1

    client = IntervalsClient(settings, sleep=concurrent_sleep)
    client._last_request_at = 0.0

    await asyncio.gather(client._throttle(), client._throttle())

    assert max_active == 2, "the lock must be released before sleeping"
    await client.aclose()


def test_the_pacing_interval_follows_the_remaining_quota(settings: Settings) -> None:
    client, _ = make_client(settings, [])
    base = 1.0 / settings.requests_per_second
    cases = [
        (RateLimits(), base),
        (RateLimits(limit_15m=100, remaining_15m=50), base),
        (RateLimits(limit_15m=100, remaining_15m=20), base * 2),
        (RateLimits(limit_15m=100, remaining_15m=10), base * 4),
        (RateLimits(limit_15m=100, remaining_15m=0), base * 4),
        (RateLimits(limit_15m=100, remaining_15m=150), base),
        (RateLimits(limit_15m=0, remaining_15m=0), base),
        (RateLimits(limit_15m=None, remaining_15m=None), base),
        (RateLimits(limit_daily=1000, remaining_daily=100), base * 4),
        (
            RateLimits(limit_15m=100, remaining_15m=50, limit_daily=1000, remaining_daily=50),
            base * 4,
        ),
    ]
    for limits, expected in cases:
        client.rate_limits = limits
        assert client._pacing_interval() == pytest.approx(expected)


async def test_an_exhausted_quota_waits_before_the_next_call(settings: Settings) -> None:
    sleep = RecordingSleep()
    client, _ = make_client(
        settings,
        [
            httpx.Response(
                200,
                json=[],
                headers={
                    "X-RateLimit-Limit": "2500,5000",
                    "X-RateLimit-Remaining": "0,4000",
                },
            ),
            httpx.Response(200, json=[]),
        ],
        sleep=sleep,
    )

    await client.list_activities("2026-08-01", "2026-08-02")
    await client.list_activities("2026-08-01", "2026-08-02")

    assert any(delay >= RATE_LIMIT_COOLDOWN_SECONDS - 0.5 for delay in sleep.calls)
    await client.aclose()


async def test_a_healthy_quota_never_waits_the_cooldown(settings: Settings) -> None:
    sleep = RecordingSleep()
    paced = settings.model_copy(update={"requests_per_second": 2.0})
    client, _ = make_client(
        paced,
        [
            httpx.Response(
                200,
                json=[],
                headers={
                    "X-RateLimit-Limit": "2500,5000",
                    "X-RateLimit-Remaining": "2000,4000",
                },
            ),
            httpx.Response(200, json=[]),
        ],
        sleep=sleep,
    )

    await client.list_activities("2026-08-01", "2026-08-02")
    await client.list_activities("2026-08-01", "2026-08-02")

    assert len(sleep.calls) == 1
    assert sleep.calls[0] == pytest.approx(0.5, abs=0.01)
    await client.aclose()


async def test_concurrent_calls_share_one_cooldown(settings: Settings) -> None:
    sleep = RecordingSleep()
    responses = [
        httpx.Response(
            200,
            json=[],
            headers={
                "X-RateLimit-Limit": "2500,5000",
                "X-RateLimit-Remaining": "0,4000",
            },
        ),
        *[httpx.Response(200, json=[]) for _ in range(4)],
    ]
    client, _ = make_client(settings, responses, sleep=sleep)

    await asyncio.gather(*(client.list_activities("2026-08-01", "2026-08-02") for _ in range(5)))

    assert max(sleep.calls) == pytest.approx(RATE_LIMIT_COOLDOWN_SECONDS, abs=1.0)
    await client.aclose()


async def test_a_create_is_not_retried_after_an_ambiguous_failure(settings: Settings) -> None:
    attempts: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ReadError("connection reset after the request was sent")

    client = IntervalsClient(
        settings, transport=httpx.MockTransport(handler), sleep=RecordingSleep()
    )

    with pytest.raises(IntervalsApiError) as error:
        await client.create_event({"name": "Tempo"})

    assert error.value.status_code == 0
    assert len(attempts) == 1
    await client.aclose()


async def test_a_create_is_not_retried_after_a_server_error(settings: Settings) -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(503, json={"error": "unavailable"})

    client = IntervalsClient(
        settings, transport=httpx.MockTransport(handler), sleep=RecordingSleep()
    )

    with pytest.raises(IntervalsApiError) as error:
        await client.create_event({"name": "Tempo"})

    assert error.value.status_code == 503
    assert len(captured) == 1
    await client.aclose()


async def test_a_create_still_retries_after_a_rate_limit(settings: Settings) -> None:
    captured: list[httpx.Request] = []
    responses = [
        httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "slow down"}),
        httpx.Response(200, json={"id": 7}),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return responses.pop(0)

    client = IntervalsClient(
        settings, transport=httpx.MockTransport(handler), sleep=RecordingSleep()
    )

    created = await client.create_event({"name": "Tempo"})

    assert created == {"id": 7}
    assert len(captured) == 2
    await client.aclose()


class FakeMonotonic:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


async def test_a_late_caller_only_waits_the_remaining_cooldown(settings: Settings) -> None:
    clock = FakeMonotonic()
    sleep = RecordingSleep()
    client, _ = make_client(
        settings,
        [
            httpx.Response(
                200,
                json=[],
                headers={
                    "X-RateLimit-Limit": "2500,5000",
                    "X-RateLimit-Remaining": "0,4000",
                },
            ),
            httpx.Response(200, json=[]),
            httpx.Response(200, json=[]),
        ],
        sleep=sleep,
        clock=clock,
    )

    await client.list_activities("2026-08-01", "2026-08-02")
    armed = sleep.calls[-1] if sleep.calls else 0.0
    assert armed < 1.0

    clock.now = 30.0
    await client.list_activities("2026-08-01", "2026-08-02")
    assert sleep.calls[-1] == pytest.approx(30.0, abs=0.01)

    clock.now = 70.0
    before = len(sleep.calls)
    await client.list_activities("2026-08-01", "2026-08-02")
    assert all(delay < 1.0 for delay in sleep.calls[before:])
    await client.aclose()

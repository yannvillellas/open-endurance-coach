import json
from datetime import date, timedelta

import pytest

from open_endurance_coach.clients.llm import LlmClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.context import RIDE_TYPES, ContextPlan, DateRange
from open_endurance_coach.extractors.resolution import resolve_context_plan

from .fakes import FakeLlmProvider, RecordingSleep

TODAY = date(2024, 2, 1)


def make_client(settings: Settings, payloads: list[object]) -> LlmClient:
    provider = FakeLlmProvider(resolutions=[json.dumps(payload) for payload in payloads])
    return LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": provider},
        sleep=RecordingSleep(),
    )


async def test_a_resolved_window_builds_the_plan(settings: Settings) -> None:
    client = make_client(
        settings,
        [
            {
                "window": {"start": "2023-12-01", "end": "2023-12-20"},
                "sport": "Ride",
                "metric": "heart_rate",
                "detail": False,
            }
        ],
    )

    resolution = await resolve_context_plan(client, "average heart rate", today=TODAY)

    assert resolution.plan.activities == DateRange(date(2023, 12, 1), date(2023, 12, 20))
    assert resolution.plan.wellness == DateRange(date(2023, 12, 1), date(2023, 12, 20))
    assert resolution.plan.events == DateRange(date(2023, 12, 1), date(2023, 12, 20))
    assert resolution.plan.activity_types == RIDE_TYPES
    assert resolution.plan.metric_focus == "heart_rate"
    assert resolution.plan.detail is False
    assert resolution.plan.reference is None
    assert resolution.notices == (
        "Fetch: 2023-12-01 → 2023-12-20 · sport: Ride · metric: heart rate",
    )


async def test_a_null_window_keeps_the_standard_plan(settings: Settings) -> None:
    client = make_client(settings, [{"window": None}])

    resolution = await resolve_context_plan(client, "how is my week going", today=TODAY)

    assert resolution.plan == ContextPlan.standard(TODAY)
    assert resolution.notices == ("Fetch: standard windows",)


async def test_a_null_window_with_detail_keeps_the_standard_windows(settings: Settings) -> None:
    client = make_client(settings, [{"window": None, "detail": True}])

    resolution = await resolve_context_plan(client, "details of my latest ride", today=TODAY)

    assert resolution.plan.detail is True
    assert resolution.plan.reference is None
    assert resolution.plan.activities == ContextPlan.standard(TODAY).activities


async def test_a_window_ending_in_the_future_is_clamped_to_today(settings: Settings) -> None:
    client = make_client(
        settings,
        [{"window": {"start": "2024-01-25", "end": "2024-02-10"}}],
    )

    resolution = await resolve_context_plan(client, "this week so far", today=TODAY)

    assert resolution.plan.activities == DateRange(date(2024, 1, 25), TODAY)


async def test_long_windows_are_capped_and_announced(settings: Settings) -> None:
    start = TODAY - timedelta(days=4000)
    client = make_client(
        settings,
        [{"window": {"start": start.isoformat(), "end": TODAY.isoformat()}}],
    )

    resolution = await resolve_context_plan(client, "the last 4000 days", today=TODAY)

    assert resolution.plan.activities == DateRange(TODAY - timedelta(days=3650), TODAY)
    assert resolution.plan.wellness == DateRange(TODAY - timedelta(days=365), TODAY)
    assert resolution.plan.events == DateRange(
        TODAY - timedelta(days=365), TODAY + timedelta(days=14)
    )
    assert any("3651 dates" in notice for notice in resolution.notices)
    assert any("366 dates" in notice for notice in resolution.notices)


async def test_an_invalid_window_falls_back_to_standard(settings: Settings) -> None:
    client = make_client(
        settings,
        [{"window": {"start": "2024-01-20", "end": "2024-01-10"}}],
    )

    resolution = await resolve_context_plan(client, "broken window", today=TODAY)

    assert resolution.plan == ContextPlan.standard(TODAY)
    assert "invalid" in resolution.notices[0]


async def test_a_failed_resolution_falls_back_to_standard(settings: Settings) -> None:
    client = make_client(settings, ["not json", "still not json"])

    resolution = await resolve_context_plan(client, "any question", today=TODAY)

    assert resolution.plan == ContextPlan.standard(TODAY)
    assert "failed" in resolution.notices[0]


async def test_a_dated_detail_pins_the_window_end(settings: Settings) -> None:
    client = make_client(
        settings,
        [{"window": {"start": "2024-01-30", "end": "2024-01-30"}, "detail": True}],
    )

    resolution = await resolve_context_plan(client, "the splits of that ride", today=TODAY)

    assert resolution.plan.detail is True
    assert resolution.plan.reference == date(2024, 1, 30)


@pytest.mark.parametrize(
    ("sport", "expected"),
    [
        ("GravelRide", RIDE_TYPES),
        ("Ride", RIDE_TYPES),
        ("cycling", RIDE_TYPES),
        ("bike", RIDE_TYPES),
        ("Run", frozenset({"Run"})),
        ("running", frozenset({"Run"})),
        ("swimming", frozenset({"Swim"})),
    ],
)
async def test_the_sport_ranks_activities_without_filtering(
    settings: Settings, sport: str, expected: frozenset[str]
) -> None:
    client = make_client(settings, [{"window": None, "sport": sport}])

    resolution = await resolve_context_plan(client, "the question", today=TODAY)

    assert resolution.plan.activity_types == expected


async def test_the_resolution_request_carries_the_focus_and_the_detail_anchor(
    settings: Settings,
) -> None:
    provider = FakeLlmProvider(resolutions=['{"window": null}'])
    client = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": provider},
        sleep=RecordingSleep(),
    )

    await resolve_context_plan(client, "the splits of my ride on 2024-01-30", today=TODAY)

    request = provider.resolution_calls[0]
    assert request["json_mode"] is True
    assert "the splits of my ride on 2024-01-30" in request["messages"][1].content
    assert "the window end must be that activity's date" in request["messages"][0].content


async def test_declared_facts_set_the_athlete_state(settings: Settings) -> None:
    provider = FakeLlmProvider(
        resolutions=[json.dumps({"window": None, "illness": True, "elite": True})]
    )
    client = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": provider},
        sleep=RecordingSleep(),
    )

    resolution = await resolve_context_plan(
        client,
        "how should I train this week",
        today=TODAY,
        facts=["[health_medical] Flu since Monday", "[athlete_profile] Professional cyclist"],
    )

    assert resolution.plan.illness is True
    assert resolution.plan.elite is True
    assert resolution.notices == ("Fetch: standard windows · state: illness, elite",)
    request = provider.resolution_calls[0]
    assert "Flu since Monday" in request["messages"][1].content
    assert "Professional cyclist" in request["messages"][1].content
    assert "illness: true only when an active athlete fact" in request["messages"][0].content


async def test_no_declared_facts_leave_the_state_off(settings: Settings) -> None:
    client = make_client(settings, [{"window": None}])

    resolution = await resolve_context_plan(client, "the question", today=TODAY)

    assert resolution.plan.illness is False
    assert resolution.plan.elite is False
    assert "state:" not in resolution.notices[0]

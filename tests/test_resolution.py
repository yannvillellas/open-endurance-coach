import json
from datetime import date, timedelta

import pytest

from open_endurance_coach.clients.llm import LlmClient, LlmMessage
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.context import RIDE_TYPES, ContextPlan, DateRange
from open_endurance_coach.extractors.resolution import (
    _resolution_context,
    _resolution_system,
    resolve_context_plan,
)
from open_endurance_coach.extractors.standard import PLAN_HORIZON_DAYS, UPCOMING_DAYS
from open_endurance_coach.tokens import estimate_text_tokens

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

    assert resolution.plan.activities == (DateRange(date(2023, 12, 1), date(2023, 12, 20)),)
    assert resolution.plan.wellness == (DateRange(date(2023, 12, 1), date(2023, 12, 20)),)
    assert resolution.plan.events == (DateRange(date(2023, 12, 1), date(2023, 12, 20)),)
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

    assert resolution.plan.activities == (DateRange(date(2024, 1, 25), TODAY),)


async def test_long_windows_are_capped_and_announced(settings: Settings) -> None:
    start = TODAY - timedelta(days=4000)
    client = make_client(
        settings,
        [{"window": {"start": start.isoformat(), "end": TODAY.isoformat()}}],
    )

    resolution = await resolve_context_plan(client, "the last 4000 days", today=TODAY)

    assert resolution.plan.activities == (DateRange(TODAY - timedelta(days=3650), TODAY),)
    assert resolution.plan.wellness == (DateRange(TODAY - timedelta(days=365), TODAY),)
    assert resolution.plan.events == (
        DateRange(TODAY - timedelta(days=365), TODAY + timedelta(days=14)),
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


async def test_an_invalid_window_still_announces_the_declared_state(settings: Settings) -> None:
    client = make_client(
        settings,
        [
            {
                "window": {"start": "2024-01-20", "end": "2024-01-10"},
                "illness": True,
            }
        ],
    )

    resolution = await resolve_context_plan(
        client, "broken window", today=TODAY, facts=["[health_medical] Flu"]
    )

    assert resolution.plan.illness is True
    assert resolution.plan.elite is False
    assert resolution.notices[0].startswith("Fetch: standard windows")
    assert "state: illness" in resolution.notices[0]
    assert "invalid" in resolution.notices[0]


async def test_a_long_history_is_trimmed_to_the_resolution_budget(settings: Settings) -> None:
    provider = FakeLlmProvider(resolutions=[json.dumps({"window": None})])
    client = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": provider},
        sleep=RecordingSleep(),
    )
    history = [
        turn
        for index in range(20)
        for turn in (
            LlmMessage(role="user", content=f"question {index} " + "x" * 800),
            LlmMessage(role="assistant", content=f"answer {index} " + "y" * 800),
        )
    ]

    resolution = await resolve_context_plan(
        client, "short question", today=TODAY, history=history, max_tokens=600
    )

    assert resolution.plan == ContextPlan.standard(TODAY)
    request = provider.resolution_calls[0]
    total = sum(estimate_text_tokens(message.content) for message in request["messages"])
    assert total <= 600
    assert "question 19 " in request["messages"][1].content
    assert "question 0 " not in request["messages"][1].content


async def test_without_a_budget_the_resolution_keeps_the_whole_history(
    settings: Settings,
) -> None:
    provider = FakeLlmProvider(resolutions=[json.dumps({"window": None})])
    client = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": provider},
        sleep=RecordingSleep(),
    )
    history = [
        LlmMessage(role="user", content="question 0 " + "x" * 800),
        LlmMessage(role="assistant", content="answer 0 " + "y" * 800),
    ]

    await resolve_context_plan(client, "short question", today=TODAY, history=history)

    assert "question 0 " in provider.resolution_calls[0]["messages"][1].content


async def test_a_thin_budget_drops_the_history_instead_of_overshooting(
    settings: Settings,
) -> None:
    provider = FakeLlmProvider(resolutions=[json.dumps({"window": None})])
    client = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": provider},
        sleep=RecordingSleep(),
    )
    fixed = estimate_text_tokens(_resolution_system(TODAY)) + estimate_text_tokens(
        _resolution_context((), "short question")
    )
    history = [
        LlmMessage(role="user", content="q"),
        LlmMessage(role="assistant", content="a"),
    ]

    await resolve_context_plan(
        client, "short question", today=TODAY, history=history, max_tokens=fixed + 5
    )

    request = provider.resolution_calls[0]
    total = sum(estimate_text_tokens(message.content) for message in request["messages"])
    assert total <= fixed + 5
    assert "Conversation so far:" not in request["messages"][1].content


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


async def test_a_clarification_sees_the_conversation(settings: Settings) -> None:
    provider = FakeLlmProvider(
        resolutions=[json.dumps({"window": {"start": "2023-12-01", "end": "2023-12-20"}})]
    )
    client = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": provider},
        sleep=RecordingSleep(),
    )
    history = [
        LlmMessage(role="user", content="what was my average hr 1-20 dec 2023"),
        LlmMessage(role="assistant", content="Your average was 159 bpm."),
    ]

    resolution = await resolve_context_plan(
        client, "sorry I meant resting hr", today=TODAY, history=history
    )

    assert resolution.plan.activities == (DateRange(date(2023, 12, 1), date(2023, 12, 20)),)
    request = provider.resolution_calls[0]
    assert "Conversation so far:" in request["messages"][1].content
    assert "what was my average hr 1-20 dec 2023" in request["messages"][1].content
    assert (
        "A follow-up or clarification keeps the period of the question it refers to."
        in request["messages"][0].content
    )


async def test_a_review_signal_resolves_the_planning_horizon(settings: Settings) -> None:
    client = make_client(settings, [{"window": None, "review_execution": True}])

    resolution = await resolve_context_plan(
        client, "how did my weeks go against the plan?", today=TODAY
    )

    assert resolution.review_execution is True
    assert resolution.review_default is True
    horizon = DateRange(TODAY - timedelta(days=PLAN_HORIZON_DAYS), TODAY)
    assert resolution.plan.activities == (horizon,)
    assert resolution.plan.events == (
        DateRange(TODAY - timedelta(days=PLAN_HORIZON_DAYS), TODAY + timedelta(days=UPCOMING_DAYS)),
    )
    assert resolution.plan.wellness == ContextPlan.standard(TODAY).wellness
    assert resolution.notices[0].startswith("Fetch: 2023-06-22 → 2024-02-01")


async def test_a_review_with_an_explicit_period_keeps_the_window(settings: Settings) -> None:
    client = make_client(
        settings,
        [
            {
                "window": {"start": "2023-12-01", "end": "2023-12-20"},
                "review_execution": True,
            }
        ],
    )

    resolution = await resolve_context_plan(client, "review 1-20 dec against the plan", today=TODAY)

    assert resolution.review_execution is True
    assert resolution.review_default is False
    assert resolution.plan.activities == (DateRange(date(2023, 12, 1), date(2023, 12, 20)),)
    assert resolution.plan.events == (DateRange(date(2023, 12, 1), date(2023, 12, 20)),)


async def test_review_execution_defaults_to_false(settings: Settings) -> None:
    client = make_client(settings, [{"window": None}])

    resolution = await resolve_context_plan(client, "status check", today=TODAY)

    assert resolution.review_execution is False


async def test_an_invalid_window_review_gets_the_default_horizon(settings: Settings) -> None:
    client = make_client(
        settings,
        [{"window": {"start": "2024-01-20", "end": "2024-01-10"}, "review_execution": True}],
    )

    resolution = await resolve_context_plan(client, "review my block", today=TODAY)

    assert resolution.review_execution is True
    assert resolution.review_default is True
    horizon = DateRange(TODAY - timedelta(days=PLAN_HORIZON_DAYS), TODAY)
    assert resolution.plan.activities == (horizon,)
    assert resolution.plan.events == (
        DateRange(TODAY - timedelta(days=PLAN_HORIZON_DAYS), TODAY + timedelta(days=UPCOMING_DAYS)),
    )
    assert "invalid window ignored" in resolution.notices[0]
    assert "Fetch: 2023-06-22 → 2024-02-01" in resolution.notices[0]


async def test_the_resolution_prompt_asks_for_the_review_signal(settings: Settings) -> None:
    provider = FakeLlmProvider(resolutions=['{"window": null}'])
    client = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": provider},
        sleep=RecordingSleep(),
    )

    await resolve_context_plan(client, "status check", today=TODAY)

    system = provider.resolution_calls[0]["messages"][0].content
    assert '"review_execution": true or false' in system
    assert "compare the past execution against what was planned" in system


async def test_a_review_keeps_the_detail_reference_and_the_sport_filter(
    settings: Settings,
) -> None:
    client = make_client(
        settings,
        [
            {
                "window": {"start": "2023-12-01", "end": "2023-12-20"},
                "review_execution": True,
                "sport": "Ride",
                "metric": "heart_rate",
                "detail": True,
            }
        ],
    )

    resolution = await resolve_context_plan(
        client, "review that ride against the plan", today=TODAY
    )

    assert resolution.plan.detail is True
    assert resolution.plan.reference == date(2023, 12, 20)
    assert resolution.plan.activity_types == RIDE_TYPES
    assert resolution.plan.metric_focus == "heart_rate"


async def test_a_default_review_keeps_the_sport_filter(settings: Settings) -> None:
    client = make_client(
        settings,
        [{"window": None, "review_execution": True, "sport": "Ride", "metric": "elevation"}],
    )

    resolution = await resolve_context_plan(
        client, "how did my climbing go against the plan?", today=TODAY
    )

    assert resolution.plan.activities == (
        DateRange(TODAY - timedelta(days=PLAN_HORIZON_DAYS), TODAY),
    )
    assert resolution.plan.activity_types == RIDE_TYPES
    assert resolution.plan.metric_focus == "elevation"


async def test_an_invalid_review_window_keeps_the_state_and_the_wellness(
    settings: Settings,
) -> None:
    client = make_client(
        settings,
        [
            {
                "window": {"start": "2024-01-20", "end": "2024-01-10"},
                "review_execution": True,
                "illness": True,
            }
        ],
    )

    resolution = await resolve_context_plan(client, "review my block while I am ill", today=TODAY)

    assert resolution.review_execution is True
    assert resolution.plan.illness is True
    horizon = DateRange(TODAY - timedelta(days=PLAN_HORIZON_DAYS), TODAY)
    assert resolution.plan.activities == (horizon,)
    assert resolution.plan.wellness == ContextPlan.standard(TODAY).wellness
    assert "state: illness" in resolution.notices[0]


async def test_a_failed_resolution_does_not_flag_a_review(settings: Settings) -> None:
    client = make_client(settings, ["not json", "still not json"])

    resolution = await resolve_context_plan(client, "review my block", today=TODAY)

    assert resolution.review_execution is False
    assert resolution.plan == ContextPlan.standard(TODAY)
    assert "resolution failed" in resolution.notices[0]

from datetime import date

import pytest

from open_endurance_coach.config import Settings
from open_endurance_coach.extractors import context as context_module
from open_endurance_coach.extractors import deep as deep_module
from open_endurance_coach.extractors.context import (
    ContextPlan,
    DateRange,
    StandardExtractor,
)
from open_endurance_coach.extractors.deep import DeepHistoricalExtractor, detect_deep_query
from open_endurance_coach.extractors.standard import (
    ACTIVITY_LOOKBACK_DAYS,
    RACE_HORIZON_DAYS,
    RECENT_EVENT_DAYS,
    ROLLUP_LOOKBACK_DAYS,
    UPCOMING_DAYS,
    WELLNESS_LOOKBACK_DAYS,
)

from .fakes import make_event, make_intervals_client, make_wellness

TODAY = date(2024, 2, 1)


def test_standard_plan_windows() -> None:
    plan = ContextPlan.standard(TODAY)

    assert (TODAY - plan.activities.start).days == ACTIVITY_LOOKBACK_DAYS
    assert (plan.activities.end - TODAY).days == 1
    assert (TODAY - plan.wellness.start).days == WELLNESS_LOOKBACK_DAYS
    assert (plan.wellness.end - TODAY).days == 1
    assert (TODAY - plan.events.start).days == RECENT_EVENT_DAYS
    assert (plan.events.end - TODAY).days == UPCOMING_DAYS
    assert plan.races.start == TODAY
    assert (plan.races.end - TODAY).days == RACE_HORIZON_DAYS
    assert (TODAY - plan.rollup.start).days == ROLLUP_LOOKBACK_DAYS
    assert plan.rollup.end == TODAY
    assert plan.activity_types == frozenset()
    assert plan.metric_focus is None
    assert plan.reference is None
    assert plan.detail is False


def test_for_activities_overrides_only_the_activity_selection() -> None:
    window = DateRange(date(2023, 11, 3), date(2024, 1, 20))
    plan = ContextPlan.standard(TODAY).for_activities(
        window=window,
        activity_types=frozenset({"Ride"}),
        metric_focus="heart_rate",
        reference=date(2024, 1, 17),
        detail=True,
    )

    assert plan.activities == window
    assert plan.activity_types == frozenset({"Ride"})
    assert plan.metric_focus == "heart_rate"
    assert plan.reference == date(2024, 1, 17)
    assert plan.detail is True
    assert plan.wellness == ContextPlan.standard(TODAY).wellness
    assert plan.events == ContextPlan.standard(TODAY).events
    assert plan.races == ContextPlan.standard(TODAY).races
    assert plan.rollup == ContextPlan.standard(TODAY).rollup


def test_with_windows_replaces_only_the_question_windows() -> None:
    plan = ContextPlan.standard(TODAY).with_windows(
        wellness=DateRange(date(2024, 1, 14), date(2024, 1, 20)),
        events=DateRange(date(2024, 1, 14), date(2024, 2, 15)),
    )

    assert plan.wellness == DateRange(date(2024, 1, 14), date(2024, 1, 20))
    assert plan.events == DateRange(date(2024, 1, 14), date(2024, 2, 15))
    assert plan.activities == ContextPlan.standard(TODAY).activities


@pytest.mark.parametrize(
    "focus",
    [
        "review the week 2 weeks ago",
        "quelle était ma charge la semaine il y a 2 semaines",
    ],
)
def test_deep_query_targets_a_named_past_week(focus: str) -> None:
    query = detect_deep_query(focus, today=TODAY)

    assert query is not None
    assert query.window == DateRange(date(2024, 1, 15), date(2024, 1, 21))
    assert query.focus_window == query.window
    assert query.reference is None


@pytest.mark.parametrize(
    "focus",
    [
        "average resting heart rate over the last 2 weeks",
        "moyenne de ma fréquence cardiaque sur les 2 dernières semaines",
    ],
)
def test_a_duration_opens_the_question_window(focus: str) -> None:
    query = detect_deep_query(focus, today=TODAY)

    assert query is not None
    assert query.window == DateRange(date(2024, 1, 18), date(2024, 2, 2))
    assert query.focus_window == query.window


def test_an_ago_point_pins_a_small_window() -> None:
    query = detect_deep_query("what was my resting heart rate 15 days ago", today=TODAY)

    assert query is not None
    assert query.reference == date(2024, 1, 17)
    assert query.focus_window == DateRange(date(2024, 1, 14), date(2024, 1, 20))
    assert query.window == DateRange(date(2023, 11, 3), date(2024, 2, 2))


async def test_a_wellness_row_from_fifteen_days_ago_is_available(
    settings: Settings,
) -> None:
    client = make_intervals_client(wellness=[make_wellness(17)])
    focus = "what was my resting heart rate 15 days ago"
    query = detect_deep_query(focus, today=TODAY)
    assert query is not None

    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=query, today=TODAY
    )

    assert ("wellness", "2024-01-14", "2024-01-20") in client.calls
    assert [row.id for row in context.wellness] == [date(2024, 1, 17)]


async def test_an_event_from_fifteen_days_ago_is_visible(settings: Settings) -> None:
    client = make_intervals_client(events=[make_event(7, "2024-01-17", name="Old session")])
    focus = "what was on my calendar 15 days ago"
    query = detect_deep_query(focus, today=TODAY)
    assert query is not None

    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=query, today=TODAY
    )

    assert [event.name for event in context.recent_events] == ["Old session"]
    events_call = next(call for call in client.calls if call[0] == "events")
    assert events_call[1] <= "2024-01-17" <= events_call[2]


async def test_a_two_week_average_gets_the_wellness_rows(settings: Settings) -> None:
    rows = [make_wellness(day) for day in range(18, 32)]
    client = make_intervals_client(wellness=rows)
    focus = "average resting heart rate over the last 2 weeks"
    query = detect_deep_query(focus, today=TODAY)
    assert query is not None

    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=query, today=TODAY
    )

    assert ("wellness", "2024-01-18", "2024-02-02") in client.calls
    assert len(context.wellness) == 14


async def test_long_question_windows_are_capped(settings: Settings) -> None:
    client = make_intervals_client(wellness=[make_wellness(17)])
    focus = "average resting heart rate over the last 60 months"
    query = detect_deep_query(focus, today=TODAY)
    assert query is not None

    await DeepHistoricalExtractor(settings, client).extract(focus, query=query, today=TODAY)

    wellness_call = next(call for call in client.calls if call[0] == "wellness")
    wellness_start = date.fromisoformat(wellness_call[1])
    assert (date.fromisoformat(wellness_call[2]) - wellness_start).days == 365
    events_call = next(call for call in client.calls if call[0] == "events")
    assert (date.fromisoformat(events_call[2]) - date.fromisoformat(events_call[1])).days <= 380


async def test_both_extractors_feed_the_single_gather(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans: list[ContextPlan] = []

    async def fake_gather(
        client: object, plan: ContextPlan, focus: str, **kwargs: object
    ) -> object:
        plans.append(plan)
        return None

    monkeypatch.setattr(context_module, "gather_context", fake_gather)
    monkeypatch.setattr(deep_module, "gather_context", fake_gather)
    client = make_intervals_client()

    await StandardExtractor(settings, client).extract("status check", today=TODAY)

    focus = "how has my heart rate trend over the last 3 months"
    query = detect_deep_query(focus, today=TODAY)
    assert query is not None
    await DeepHistoricalExtractor(settings, client).extract(focus, query=query, today=TODAY)

    assert plans[0] == ContextPlan.standard(TODAY)
    assert plans[1].activities == DateRange(date(2023, 11, 3), date(2024, 2, 2))
    assert plans[1].wellness == DateRange(date(2023, 11, 3), date(2024, 2, 2))
    assert plans[1].metric_focus == "heart_rate"
    assert plans[1].detail is True

from datetime import date

import pytest

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.extractors.context import ContextPlan, DateRange, gather_context
from open_endurance_coach.extractors.standard import (
    ACTIVITY_LOOKBACK_DAYS,
    RACE_HORIZON_DAYS,
    RECENT_EVENT_DAYS,
    ROLLUP_LOOKBACK_DAYS,
    UPCOMING_DAYS,
    WELLNESS_LOOKBACK_DAYS,
)

from .fakes import make_activity, make_intervals_client

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
    "expected",
    [
        ("activities", "2024-01-18", "2024-02-02"),
        ("wellness", "2024-01-25", "2024-02-02"),
        ("events", "2024-01-18", "2024-02-15", None),
        ("athlete_summary", "2023-11-03", "2024-02-01"),
        ("events", "2024-02-01", "2024-05-31", "RACE_A,RACE_B,RACE_C"),
        ("sport_settings",),
    ],
)
async def test_the_standard_plan_reproduces_the_tactical_calls(
    expected: tuple[object, ...],
) -> None:
    client = make_intervals_client()

    await gather_context(client, ContextPlan.standard(TODAY), "status check", today=TODAY)

    assert expected in client.calls
    assert len(client.calls) == 6


async def test_a_question_window_drives_every_source() -> None:
    window = DateRange(date(2023, 12, 1), date(2023, 12, 20))
    plan = (
        ContextPlan.standard(TODAY)
        .for_activities(window=window)
        .with_windows(wellness=window, events=window)
    )
    client = make_intervals_client()

    await gather_context(client, plan, "average heart rate", today=TODAY)

    assert ("activities", "2023-12-01", "2023-12-20") in client.calls
    assert ("wellness", "2023-12-01", "2023-12-20") in client.calls
    assert ("events", "2023-12-01", "2023-12-20", None) in client.calls


async def test_a_detail_plan_pins_the_referenced_activity() -> None:
    client = make_intervals_client(activities=[make_activity("fx-old", 30)])
    plan = ContextPlan.standard(TODAY).for_activities(
        window=DateRange(date(2024, 1, 30), date(2024, 1, 30)),
        reference=date(2024, 1, 30),
        detail=True,
    )

    context = await gather_context(client, plan, "the splits of that ride", today=TODAY)

    assert context.activity_detail is not None
    assert context.activity_detail.id == "fx-old"
    assert any(call[0] == "streams" and call[1] == "fx-old" for call in client.calls)


async def test_a_detail_plan_includes_the_splits() -> None:
    streams = {
        "time": list(range(601)),
        "distance": [index * 1000 / 300 for index in range(601)],
        "heartrate": [150] * 601,
    }
    client = make_intervals_client(streams=streams)
    plan = ContextPlan.standard(TODAY).for_activities(
        window=DateRange(date(2024, 1, 18), date(2024, 1, 20)),
        reference=date(2024, 1, 20),
        detail=True,
    )

    context = await gather_context(client, plan, "the splits", today=TODAY)

    assert [split.label for split in context.activity_splits] == ["km 1", "km 2"]
    assert context.activity_splits[0].average_heartrate == 150
    assert any(call[0] == "streams" and "watts" in call[2] for call in client.calls)


async def test_a_streams_failure_keeps_the_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_intervals_client()

    async def boom(activity_id: str, types: object) -> dict[str, list]:
        raise IntervalsApiError(404, "no streams")

    monkeypatch.setattr(client, "get_activity_streams", boom)
    plan = ContextPlan.standard(TODAY).for_activities(
        window=DateRange(date(2024, 1, 18), date(2024, 1, 20)),
        reference=date(2024, 1, 20),
        detail=True,
    )

    context = await gather_context(client, plan, "the splits", today=TODAY)

    assert context.activity_detail is not None
    assert context.activity_splits == []


async def test_a_named_sport_ranks_first_without_dropping() -> None:
    client = make_intervals_client(
        activities=[
            make_activity("fx-run", 20, activity_type="Run"),
            make_activity("fx-ride", 18),
        ]
    )
    plan = ContextPlan.standard(TODAY).for_activities(
        window=DateRange(date(2024, 1, 1), date(2024, 2, 2)),
        activity_types=frozenset({"Ride"}),
    )

    context = await gather_context(client, plan, "my rides", today=TODAY)

    assert [activity.id for activity in context.recent_activities] == ["fx-ride", "fx-run"]

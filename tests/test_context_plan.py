from dataclasses import replace
from datetime import date, timedelta

import pytest

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.extractors import context as context_module
from open_endurance_coach.extractors.context import ContextPlan, DateRange, gather_context
from open_endurance_coach.extractors.standard import (
    ACTIVITY_LOOKBACK_DAYS,
    PLAN_HORIZON_DAYS,
    RECENT_EVENT_DAYS,
    UPCOMING_DAYS,
    WELLNESS_LOOKBACK_DAYS,
)

from .fakes import (
    FakeIntervalsClient,
    make_activity,
    make_intervals_client,
    make_summary_week,
    make_wellness,
)

TODAY = date(2024, 2, 1)


def test_standard_plan_windows() -> None:
    plan = ContextPlan.standard(TODAY)

    assert (TODAY - plan.activities[0].start).days == ACTIVITY_LOOKBACK_DAYS
    assert (plan.activities[0].end - TODAY).days == 1
    assert (TODAY - plan.wellness[0].start).days == WELLNESS_LOOKBACK_DAYS
    assert (plan.wellness[0].end - TODAY).days == 1
    assert (TODAY - plan.events[0].start).days == RECENT_EVENT_DAYS
    assert (plan.events[0].end - TODAY).days == UPCOMING_DAYS
    assert plan.races.start == TODAY
    assert (plan.races.end - TODAY).days == PLAN_HORIZON_DAYS
    assert (TODAY - plan.rollup.start).days == PLAN_HORIZON_DAYS
    assert plan.rollup.end == TODAY
    assert plan.activity_types == frozenset()
    assert plan.metric_focus is None
    assert plan.reference is None
    assert plan.detail is False


def test_for_activities_overrides_only_the_activity_selection() -> None:
    window = DateRange(date(2023, 11, 3), date(2024, 1, 20))
    plan = ContextPlan.standard(TODAY).for_activities(
        windows=(window,),
        activity_types=frozenset({"Ride"}),
        metric_focus="heart_rate",
        reference=date(2024, 1, 17),
        detail=True,
    )

    assert plan.activities == (window,)
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
        wellness=(DateRange(date(2024, 1, 14), date(2024, 1, 20)),),
        events=(DateRange(date(2024, 1, 14), date(2024, 2, 15)),),
    )

    assert plan.wellness == (DateRange(date(2024, 1, 14), date(2024, 1, 20)),)
    assert plan.events == (DateRange(date(2024, 1, 14), date(2024, 2, 15)),)
    assert plan.activities == ContextPlan.standard(TODAY).activities


@pytest.mark.parametrize(
    "expected",
    [
        ("activities", "2024-01-18", "2024-02-02"),
        ("wellness", "2024-01-25", "2024-02-02"),
        ("events", "2024-01-18", "2024-02-15", None),
        (
            "athlete_summary",
            (TODAY - timedelta(days=PLAN_HORIZON_DAYS)).isoformat(),
            "2024-02-01",
        ),
        (
            "events",
            "2024-02-01",
            (TODAY + timedelta(days=PLAN_HORIZON_DAYS)).isoformat(),
            "RACE_A,RACE_B,RACE_C",
        ),
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
        .for_activities(windows=(window,))
        .with_windows(wellness=(window,), events=(window,))
    )
    client = make_intervals_client()

    await gather_context(client, plan, "average heart rate", today=TODAY)

    assert ("activities", "2023-12-01", "2023-12-20") in client.calls
    assert ("wellness", "2023-12-01", "2023-12-20") in client.calls
    assert ("events", "2023-12-01", "2023-12-20", None) in client.calls


async def test_a_detail_plan_pins_the_referenced_activity() -> None:
    client = make_intervals_client(activities=[make_activity("fx-old", 30)])
    plan = ContextPlan.standard(TODAY).for_activities(
        windows=(DateRange(date(2024, 1, 30), date(2024, 1, 30)),),
        reference=date(2024, 1, 30),
        detail=True,
    )

    context = await gather_context(client, plan, "the splits of that ride", today=TODAY)

    assert context.activity_detail is not None
    assert context.activity_detail.id == "fx-old"
    assert any(call[0] == "streams" and call[1] == "fx-old" for call in client.calls)


async def test_the_plan_state_reaches_the_block_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, bool] = {}
    real = context_module.derive_block_plan

    def capture(
        races: object, weeks: object, *, today: date, illness: bool = False, elite: bool = False
    ) -> object:
        captured["illness"] = illness
        captured["elite"] = elite
        return real(races, weeks, today=today, illness=illness, elite=elite)  # type: ignore[arg-type]

    monkeypatch.setattr(context_module, "derive_block_plan", capture)
    plan = replace(ContextPlan.standard(TODAY), illness=True, elite=True)

    await gather_context(make_intervals_client(), plan, "focus", today=TODAY)

    assert captured == {"illness": True, "elite": True}


async def test_a_detail_plan_includes_the_splits() -> None:
    streams = {
        "time": list(range(601)),
        "distance": [index * 1000 / 300 for index in range(601)],
        "heartrate": [150] * 601,
    }
    client = make_intervals_client(streams=streams)
    plan = ContextPlan.standard(TODAY).for_activities(
        windows=(DateRange(date(2024, 1, 18), date(2024, 1, 20)),),
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
        windows=(DateRange(date(2024, 1, 18), date(2024, 1, 20)),),
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
        windows=(DateRange(date(2024, 1, 1), date(2024, 2, 2)),),
        activity_types=frozenset({"Ride"}),
    )

    context = await gather_context(client, plan, "my rides", today=TODAY)

    assert [activity.id for activity in context.recent_activities] == ["fx-ride", "fx-run"]


async def test_the_detail_pin_prefers_the_named_sport_then_the_nearest() -> None:
    client = make_intervals_client(
        activities=[
            make_activity("fx-run", 20, activity_type="Run"),
            make_activity("fx-ride", 18),
        ]
    )
    plan = ContextPlan.standard(TODAY).for_activities(
        windows=(DateRange(date(2024, 1, 15), date(2024, 1, 23)),),
        reference=date(2024, 1, 20),
        activity_types=frozenset({"Ride"}),
        detail=True,
    )

    context = await gather_context(client, plan, "the ride", today=TODAY)

    assert context.activity_detail is not None
    assert context.activity_detail.id == "fx-ride"


async def test_the_detail_pin_falls_back_to_any_sport() -> None:
    client = make_intervals_client(activities=[make_activity("fx-run", 20, activity_type="Run")])
    plan = ContextPlan.standard(TODAY).for_activities(
        windows=(DateRange(date(2024, 1, 15), date(2024, 1, 23)),),
        reference=date(2024, 1, 20),
        activity_types=frozenset({"Ride"}),
        detail=True,
    )

    context = await gather_context(client, plan, "the ride", today=TODAY)

    assert context.activity_detail is not None
    assert context.activity_detail.id == "fx-run"


async def test_a_detail_failure_keeps_the_context(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_intervals_client()

    async def boom(activity_id: str, intervals: bool = True) -> dict:
        raise IntervalsApiError(500, "boom")

    monkeypatch.setattr(client, "get_activity", boom)
    plan = ContextPlan.standard(TODAY).for_activities(
        windows=(DateRange(date(2024, 1, 18), date(2024, 1, 20)),),
        reference=date(2024, 1, 20),
        detail=True,
    )

    context = await gather_context(client, plan, "the detail", today=TODAY)

    assert context.activity_detail is None
    assert context.activity_splits == []


@pytest.mark.parametrize("error", [ValueError("bad streams"), RuntimeError("unexpected")])
async def test_unexpected_stream_errors_keep_the_detail(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    client = make_intervals_client()

    async def boom(activity_id: str, types: object) -> dict[str, list]:
        raise error

    monkeypatch.setattr(client, "get_activity_streams", boom)
    plan = ContextPlan.standard(TODAY).for_activities(
        windows=(DateRange(date(2024, 1, 18), date(2024, 1, 20)),),
        reference=date(2024, 1, 20),
        detail=True,
    )

    context = await gather_context(client, plan, "the detail", today=TODAY)

    assert context.activity_detail is not None
    assert context.activity_splits == []


def _review_client() -> FakeIntervalsClient:
    weeks = [date(2024, 1, 1) + timedelta(days=7 * index) for index in range(7)]
    activities = [
        {
            "id": f"a-{day.isoformat()}",
            "start_date_local": f"{day.isoformat()}T08:00:00",
            "type": "Ride",
            "name": f"Ride {day.isoformat()}",
            "icu_training_load": 100.0,
            "moving_time": 3600,
        }
        for day in weeks[:-1]
        if day != date(2024, 1, 29)
    ]
    activities.append(
        {
            "id": "a-2024-02-06",
            "start_date_local": "2024-02-06T08:00:00",
            "type": "Run",
            "name": "Lunch run",
            "icu_training_load": 40.0,
            "moving_time": 1800,
        }
    )
    events = [
        {
            "id": index + 1,
            "name": f"Workout {index + 1}",
            "start_date_local": f"{day.isoformat()}T00:00:00",
            "category": "WORKOUT",
            "type": "Ride",
            "icu_training_load": 100.0,
        }
        for index, day in enumerate(weeks[:-1])
    ]
    events.append(
        {
            "id": 99,
            "name": "Summer Race",
            "start_date_local": "2024-07-08T00:00:00",
            "category": "RACE_A",
            "type": "Run",
            "icu_training_load": 300.0,
        }
    )
    summary = [
        make_summary_week(day.isoformat(), fitness=50.0 + index, ramp_rate=2.0)
        for index, day in enumerate(weeks)
    ]
    return make_intervals_client(activities=activities, events=events, athlete_summary=summary)


async def test_gather_context_builds_the_execution_review() -> None:
    today = date(2024, 2, 12)
    plan = replace(
        ContextPlan.standard(today),
        activities=(DateRange(today - timedelta(days=45), today),),
        events=(DateRange(today - timedelta(days=45), today + timedelta(days=14)),),
    )

    context = await gather_context(
        _review_client(),
        plan,
        "how did my weeks go against the plan?",
        today=today,
        review_execution=True,
        review_default=True,
    )

    review = context.execution_review
    assert review is not None
    reviewed = [session.date for session in review.sessions]
    assert min(reviewed) >= date(2024, 1, 22)
    assert date(2024, 1, 1) not in reviewed
    assert date(2024, 1, 15) not in reviewed
    assert any(session.status == "missed" for session in review.sessions)
    assert any(session.status == "unplanned" for session in review.sessions)
    assert review.weeks[0].week_start == date(2024, 1, 22)
    assert review.weeks[-1].current is True


async def test_gather_context_skips_the_review_without_the_signal() -> None:
    today = date(2024, 2, 12)
    plan = replace(
        ContextPlan.standard(today),
        activities=(DateRange(today - timedelta(days=45), today),),
        events=(DateRange(today - timedelta(days=45), today + timedelta(days=14)),),
    )

    context = await gather_context(_review_client(), plan, "status check", today=today)

    assert context.execution_review is None


async def test_an_explicit_review_period_is_not_clipped_at_the_block() -> None:
    today = date(2024, 2, 12)
    plan = replace(
        ContextPlan.standard(today),
        activities=(DateRange(today - timedelta(days=60), today),),
        events=(DateRange(today - timedelta(days=60), today + timedelta(days=14)),),
    )

    context = await gather_context(
        _review_client(),
        plan,
        "review January against the plan",
        today=today,
        review_execution=True,
    )

    review = context.execution_review
    assert review is not None
    assert date(2024, 1, 1) in [session.date for session in review.sessions]
    assert review.weeks[0].week_start == date(2024, 1, 22)


async def test_a_review_without_a_race_reads_no_digest() -> None:
    today = date(2024, 2, 12)
    client = make_intervals_client(
        athlete_summary=[make_summary_week("2024-02-05"), make_summary_week("2024-02-12")]
    )
    plan = replace(
        ContextPlan.standard(today),
        activities=(DateRange(today - timedelta(days=45), today),),
        events=(DateRange(today - timedelta(days=45), today + timedelta(days=14)),),
    )

    context = await gather_context(
        client, plan, "review my block", today=today, review_execution=True, review_default=True
    )

    assert context.execution_review is None


async def test_the_digest_is_computed_before_the_budget_eviction() -> None:
    today = date(2024, 2, 12)
    plan = replace(
        ContextPlan.standard(today),
        activities=(DateRange(today - timedelta(days=45), today),),
        events=(DateRange(today - timedelta(days=45), today + timedelta(days=14)),),
    )

    context = await gather_context(
        _review_client(),
        plan,
        "how did my weeks go against the plan?",
        today=today,
        max_tokens=1200,
        review_execution=True,
        review_default=True,
    )

    assert context.execution_review is not None
    oldest = date(2024, 1, 22)
    assert oldest in [session.date for session in context.execution_review.sessions]
    assert all(activity.start_date_local.date() > oldest for activity in context.recent_activities)


async def test_multiple_windows_fetch_each_and_merge_by_id() -> None:
    early = DateRange(date(2024, 1, 1), date(2024, 1, 10))
    late = DateRange(date(2024, 1, 5), date(2024, 1, 15))
    plan = replace(
        ContextPlan.standard(TODAY),
        activities=(early, late),
        wellness=(early, late),
        events=(early, late),
    )
    client = make_intervals_client(
        activities=[make_activity("fx-a", 3), make_activity("fx-b", 7), make_activity("fx-c", 12)],
        wellness=[make_wellness(3), make_wellness(7), make_wellness(12)],
        events=[
            {"id": 1, "name": "A", "start_date_local": "2024-01-03T00:00:00"},
            {"id": 2, "name": "B", "start_date_local": "2024-01-12T00:00:00"},
            {"id": 3, "name": "C", "start_date_local": "2024-01-07T00:00:00"},
            {"name": "D", "start_date_local": "2024-01-04T00:00:00"},
            {"name": "D", "start_date_local": "2024-01-04T00:00:00"},
        ],
    )

    context = await gather_context(client, plan, "both windows", today=TODAY)

    assert [call for call in client.calls if call[0] == "activities"] == [
        ("activities", "2024-01-01", "2024-01-10"),
        ("activities", "2024-01-05", "2024-01-15"),
    ]
    assert [call for call in client.calls if call[0] == "wellness"] == [
        ("wellness", "2024-01-01", "2024-01-10"),
        ("wellness", "2024-01-05", "2024-01-15"),
    ]
    assert [call for call in client.calls if call[0] == "events" and call[3] is None] == [
        ("events", "2024-01-01", "2024-01-10", None),
        ("events", "2024-01-05", "2024-01-15", None),
    ]
    # The overlapping day (7) is fetched twice and kept once.
    assert [activity.id for activity in context.recent_activities] == ["fx-c", "fx-b", "fx-a"]
    assert [row.id for row in context.wellness] == [
        date(2024, 1, 12),
        date(2024, 1, 7),
        date(2024, 1, 3),
    ]
    # The id-bearing overlap (C, day 7) is kept once; the id-less duplicate (D)
    # collapses on its fields; the order stays chronological.
    assert [event.name for event in context.recent_events] == ["B", "C", "D", "A"]


async def test_empty_windows_fetch_nothing() -> None:
    plan = replace(ContextPlan.standard(TODAY), activities=(), wellness=(), events=())
    client = make_intervals_client()

    context = await gather_context(client, plan, "nothing", today=TODAY)

    assert not [
        call
        for call in client.calls
        if call[0] in ("activities", "wellness") or (call[0] == "events" and call[3] is None)
    ]
    assert context.recent_activities == []
    assert context.wellness == []
    assert context.recent_events == []

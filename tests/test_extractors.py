import logging
from datetime import date, timedelta

import pytest

from open_endurance_coach.config import Settings
from open_endurance_coach.errors import InternalError
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.extractors.context import (
    ContextPlan,
    _warn_on_split_coverage,
    gather_context,
)
from open_endurance_coach.extractors.standard import PLAN_HORIZON_DAYS, training_rollup
from open_endurance_coach.schemas.context import (
    BlockWeek,
    CoachContext,
    ExecutionReview,
    GoalRace,
    ReviewWeek,
    TrainingWeek,
)
from open_endurance_coach.schemas.intervals import (
    Activity,
    ActivitySplit,
    Event,
    SportSettings,
    Wellness,
)

from .fakes import (
    TODAY,
    FakeIntervalsClient,
    make_activity,
    make_event,
    make_intervals_client,
    make_summary_week,
    make_wellness,
)


async def test_standard_extraction_populates_all_sections(settings: Settings) -> None:
    context = await gather_context(
        make_intervals_client(), ContextPlan.standard(TODAY), "Analyze this week", today=TODAY
    )
    assert len(context.recent_activities) == 5
    assert len(context.wellness) == 3
    assert len(context.upcoming_events) == 2
    assert context.sport_settings[0].ftp == 250.0
    assert context.activity_detail is None
    assert context.user_feedback is None
    assert context.today == TODAY


async def test_standard_extraction_uses_expected_windows(settings: Settings) -> None:
    client = make_intervals_client()
    await gather_context(
        client, ContextPlan.standard(TODAY), "status check", user_feedback="felt tired", today=TODAY
    )
    calls = [call for call in client.calls if call[0] != "events"]
    assert calls == [
        ("activities", "2024-01-18", "2024-02-02"),
        ("wellness", "2024-01-25", "2024-02-02"),
        (
            "athlete_summary",
            (TODAY - timedelta(days=PLAN_HORIZON_DAYS)).isoformat(),
            "2024-02-01",
        ),
        ("sport_settings",),
    ]


async def test_fake_list_events_filters_by_window_and_category() -> None:
    client = FakeIntervalsClient(
        activities=[],
        wellness=[],
        sport_settings=[],
        events=[
            make_event(1, "2024-01-17", name="Too old"),
            make_event(2, "2024-01-18", name="Oldest inside"),
            make_event(3, "2024-02-15", name="Newest inside"),
            make_event(4, "2024-02-16", name="Too new"),
            make_event(5, "2024-01-20", name="Race inside", category="RACE_A"),
        ],
    )
    workouts = await client.list_events("2024-01-18", "2024-02-15", category="WORKOUT")
    assert [row["name"] for row in workouts] == ["Oldest inside", "Newest inside"]
    everything = await client.list_events("2024-01-18", "2024-02-15")
    assert [row["name"] for row in everything] == [
        "Oldest inside",
        "Newest inside",
        "Race inside",
    ]


async def test_standard_extraction_drops_events_outside_the_window(settings: Settings) -> None:
    client = make_intervals_client(
        events=[
            make_event(1, (TODAY - timedelta(days=15)).isoformat(), name="Too old"),
            make_event(2, (TODAY - timedelta(days=14)).isoformat(), name="Oldest inside"),
            make_event(3, (TODAY + timedelta(days=14)).isoformat(), name="Newest inside"),
            make_event(4, (TODAY + timedelta(days=15)).isoformat(), name="Too new"),
        ]
    )
    context = await gather_context(client, ContextPlan.standard(TODAY), "status check", today=TODAY)
    names = [event.name for event in context.recent_events + context.upcoming_events]
    assert names == ["Oldest inside", "Newest inside"]


async def test_standard_extraction_splits_past_and_upcoming_events(settings: Settings) -> None:
    client = make_intervals_client(
        events=[
            make_event(0, "2024-01-29", name="Older past session"),
            make_event(1, "2024-01-30", name="Prescribed hill session"),
            make_event(2, "2024-02-01", name="Today session"),
            make_event(3, "2024-02-03", name="Tempo Session"),
        ]
    )
    context = await gather_context(
        client, ContextPlan.standard(TODAY), "review yesterday", today=TODAY
    )
    assert [event.name for event in context.recent_events] == [
        "Prescribed hill session",
        "Older past session",
    ]
    assert [event.name for event in context.upcoming_events] == [
        "Today session",
        "Tempo Session",
    ]


async def test_budget_drops_past_events_before_upcoming(settings: Settings) -> None:
    events = [
        make_event(
            100 + index, (TODAY - timedelta(days=index + 1)).isoformat(), name=f"Past {index}"
        )
        for index in range(14)
    ] + [make_event(200, (TODAY + timedelta(days=1)).isoformat(), name="Tomorrow")]
    context = await gather_context(
        make_intervals_client(events=events),
        ContextPlan.standard(TODAY),
        "status check",
        today=TODAY,
        max_tokens=500,
    )
    assert [event.name for event in context.upcoming_events] == ["Tomorrow"]
    recent_names = [event.name for event in context.recent_events]
    assert 0 < len(recent_names) < 14
    assert recent_names == [f"Past {index}" for index in range(len(recent_names))]


def test_budget_drops_the_oldest_recent_event_regardless_of_order() -> None:
    oldest = Event.model_validate(make_event(1, "2024-01-25", name="Oldest"))
    middle = Event.model_validate(make_event(3, "2024-01-28", name="Middle"))
    newest = Event.model_validate(make_event(2, "2024-01-31", name="Newest"))

    def build(recent: list[Event], max_tokens: int) -> CoachContext:
        return build_within_budget(
            "status check",
            [],
            [],
            [],
            [],
            recent_events=recent,
            user_feedback=None,
            activity_detail=None,
            max_tokens=max_tokens,
            today=TODAY,
        )

    limit = build([middle, newest], max_tokens=10_000).data_tokens()
    context = build([middle, oldest, newest], max_tokens=limit)
    assert [event.name for event in context.recent_events] == ["Middle", "Newest"]


def test_split_coverage_warning_thresholds(caplog: pytest.LogCaptureFixture) -> None:
    splits = [ActivitySplit(label="km 1", distance_m=1000.0, time_s=300)]

    with caplog.at_level(logging.WARNING):
        _warn_on_split_coverage("a1", None, splits)
        _warn_on_split_coverage("a1", 0.0, splits)
        _warn_on_split_coverage("a1", 1000.0, splits)
        _warn_on_split_coverage("a1", 1020.0, splits)
    assert not caplog.records

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_on_split_coverage("a1", 1021.0, splits)
        _warn_on_split_coverage("a1", 1000.0, [])
    assert len(caplog.records) == 2


async def test_standard_extraction_has_no_activity_splits(settings: Settings) -> None:
    client = make_intervals_client()
    context = await gather_context(client, ContextPlan.standard(TODAY), "status check", today=TODAY)
    assert context.activity_splits == []
    assert not any(call[0] == "streams" for call in client.calls)


async def test_standard_extraction_keeps_newest_first(settings: Settings) -> None:
    context = await gather_context(
        make_intervals_client(), ContextPlan.standard(TODAY), "status check", today=TODAY
    )
    assert [item.id for item in context.recent_activities] == [
        "fx-a",
        "fx-b",
        "fx-c",
        "fx-d",
        "fx-e",
    ]


async def test_budget_overrun_trims_oldest_activities_first(settings: Settings) -> None:
    context = await gather_context(
        make_intervals_client(),
        ContextPlan.standard(TODAY),
        "status check",
        today=TODAY,
        max_tokens=250,
    )
    assert context.estimated_tokens() <= 250
    assert len(context.recent_activities) < 5
    assert [item.id for item in context.recent_activities] == [
        "fx-a",
        "fx-b",
        "fx-c",
        "fx-d",
    ][: len(context.recent_activities)]
    assert len(context.wellness) == 2
    assert len(context.upcoming_events) == 2


async def test_budget_too_small_to_fit_focus_raises(settings: Settings) -> None:
    client = make_intervals_client()
    with pytest.raises(InternalError, match="token budget"):
        await gather_context(
            client, ContextPlan.standard(TODAY), "status check", today=TODAY, max_tokens=1
        )


def test_budget_evicts_sport_settings_before_failing() -> None:
    huge = SportSettings.model_validate({"id": 1, "types": ["X" * 100_000], "ftp": 250.0})
    context = build_within_budget(
        "status check",
        [],
        [],
        [],
        [huge],
        user_feedback=None,
        activity_detail=None,
        max_tokens=500,
        today=TODAY,
    )
    assert context.sport_settings == []


def test_split_coverage_warning_stays_on_one_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _warn_on_split_coverage("a1\nFORGED", 1000.0, [])
    messages = [record.getMessage() for record in caplog.records]
    assert messages
    assert all("\n" not in message for message in messages)


def test_budget_keeps_recent_activities_and_drops_wellness_first() -> None:
    recent = [
        Activity.model_validate(make_activity("fx-near", 28)),
        Activity.model_validate(make_activity("fx-nearer", 31)),
    ]
    recent.sort(key=lambda activity: activity.start_date_local, reverse=True)
    probe = CoachContext(
        focus="status check", today=TODAY, recent_activities=recent, max_tokens=10**9
    )
    context = build_within_budget(
        focus="status check",
        recent_activities=recent,
        wellness=[Wellness.model_validate(make_wellness(28))],
        upcoming_events=[],
        sport_settings=[],
        user_feedback=None,
        activity_detail=None,
        max_tokens=probe.estimated_tokens(),
        today=TODAY,
    )
    assert len(context.recent_activities) == 2
    assert context.wellness == []


def test_budget_drops_activity_detail_before_failing() -> None:
    detail = Activity.model_validate(
        {
            **make_activity("fx-detail", 5),
            "icu_intervals": [
                {"id": index, "type": "Ride", "label": f"rep {index}", "average_watts": 300.0}
                for index in range(40)
            ],
        }
    )
    full = CoachContext(focus="f", activity_detail=detail).estimated_tokens()
    context = build_within_budget(
        focus="f",
        recent_activities=[],
        wellness=[],
        upcoming_events=[],
        sport_settings=[],
        user_feedback=None,
        activity_detail=detail,
        max_tokens=full - 5,
    )
    assert context.activity_detail is None


def test_budget_drops_the_oldest_activities_even_when_sorted_by_metric() -> None:
    activities = [
        Activity.model_validate(make_activity("fx-old-hard", 1)),
        Activity.model_validate(make_activity("fx-new-easy", 10)),
    ]
    context = build_within_budget(
        focus="f",
        recent_activities=activities,
        wellness=[],
        upcoming_events=[],
        sport_settings=[],
        user_feedback=None,
        activity_detail=None,
        max_tokens=CoachContext(focus="f", recent_activities=activities).estimated_tokens() + 1,
        today=date(2024, 2, 1),
    )
    assert [activity.id for activity in context.recent_activities] == ["fx-new-easy"]


async def test_standard_extraction_builds_goal_races(settings: Settings) -> None:
    client = make_intervals_client(
        events=[
            {
                "id": 90001,
                "name": "Spring Half",
                "start_date_local": "2024-03-01T00:00:00",
                "category": "RACE_A",
                "type": "Run",
                "moving_time": 5400,
                "distance": 21097.5,
                "icu_training_load": 150,
                "description": "Flat and fast: target 1:30, hold Z3.",
            },
            {
                "id": 90002,
                "name": "Club Crit",
                "start_date_local": "2024-02-08T00:00:00",
                "category": "RACE_B",
                "type": "Ride",
                "description": "Circuit race, 1.2 km lap, target 60 min.",
            },
            {
                "id": 90003,
                "name": "Tempo Session",
                "start_date_local": "2024-02-05T00:00:00",
                "category": "WORKOUT",
            },
        ]
    )
    context = await gather_context(client, ContextPlan.standard(TODAY), "status check", today=TODAY)
    assert [race.category for race in context.goal_races] == ["RACE_B", "RACE_A"]
    nearest, furthest = context.goal_races
    assert nearest.event_id == 90002
    assert nearest.name == "Club Crit"
    assert nearest.date == date(2024, 2, 8)
    assert nearest.type == "Ride"
    assert nearest.days_to_race == 7
    assert nearest.weeks_to_race == 1
    assert nearest.description == "Circuit race, 1.2 km lap, target 60 min."
    assert furthest.days_to_race == 29
    assert furthest.weeks_to_race == 5
    assert furthest.distance == 21097.5
    assert furthest.moving_time == 5400
    assert furthest.icu_training_load == 150
    assert furthest.description == "Flat and fast: target 1:30, hold Z3."


def test_training_rollup_is_ascending_with_partial_current_week() -> None:
    weeks = training_rollup(
        [make_summary_week("2024-01-30"), make_summary_week("2024-01-23")], today=TODAY
    )
    assert [week.week_start for week in weeks] == [date(2024, 1, 23), date(2024, 1, 30)]
    assert weeks[0].partial is False
    assert weeks[-1].partial is True
    assert weeks[-1].fitness == 30.0
    assert [sport.category for sport in weeks[-1].sports] == ["Ride", "Run"]


def test_training_rollup_zero_fills_missing_weeks() -> None:
    weeks = training_rollup(
        [make_summary_week("2024-01-30"), make_summary_week("2024-01-16")], today=TODAY
    )
    assert [week.week_start for week in weeks] == [
        date(2024, 1, 16),
        date(2024, 1, 23),
        date(2024, 1, 30),
    ]
    gap = weeks[1]
    assert (gap.sessions, gap.time_s, gap.load, gap.fitness) == (0, 0, None, None)
    assert gap.sports == []


def test_training_rollup_skips_zero_session_sports() -> None:
    weeks = training_rollup(
        [
            make_summary_week(
                "2024-01-30",
                sports=[
                    {"category": "Ride", "count": 0, "time": 0, "training_load": 0},
                    {"category": "Run", "count": 1, "time": 2400, "training_load": 30},
                ],
            )
        ],
        today=TODAY,
    )
    assert [sport.category for sport in weeks[0].sports] == ["Run"]
    assert weeks[0].sports[0].sessions == 1


def test_training_rollup_rejects_non_weekly_spacing() -> None:
    with pytest.raises(ValueError, match="bucket spacing"):
        training_rollup(
            [make_summary_week("2024-01-30"), make_summary_week("2024-01-28")], today=TODAY
        )


def test_training_rollup_returns_empty_without_rows() -> None:
    assert training_rollup([], today=TODAY) == []


def test_budget_keeps_the_newest_rollup_weeks_when_trimming() -> None:
    weeks = [
        TrainingWeek(
            week_start=date(2024, 1, 1) + timedelta(days=7 * index),
            sessions=5,
            time_s=14400,
            load=320.0,
        )
        for index in range(6)
    ]
    full = CoachContext(focus="f", training_rollup=weeks).estimated_tokens()
    context = build_within_budget(
        focus="f",
        recent_activities=[],
        wellness=[],
        upcoming_events=[],
        sport_settings=[],
        training_rollup=weeks,
        user_feedback=None,
        activity_detail=None,
        max_tokens=full - 40,
    )
    assert len(context.training_rollup) < len(weeks)
    assert context.training_rollup[-1].week_start == weeks[-1].week_start
    assert context.training_rollup[0].week_start == weeks[1].week_start


async def test_standard_extraction_includes_the_training_rollup(settings: Settings) -> None:
    context = await gather_context(
        make_intervals_client(), ContextPlan.standard(TODAY), "status check", today=TODAY
    )
    assert [week.week_start.isoformat() for week in context.training_rollup] == [
        "2024-01-22",
        "2024-01-29",
    ]
    assert [week.partial for week in context.training_rollup] == [False, True]
    assert context.training_rollup[-1].form == 2.0


async def test_past_dated_race_is_not_a_goal_race(settings: Settings) -> None:
    client = make_intervals_client(
        events=[
            {
                "id": 9001,
                "name": "Old Race",
                "start_date_local": "2024-01-04T00:00:00",
                "category": "RACE_A",
            },
            {
                "id": 9002,
                "name": "Coming Race",
                "start_date_local": "2024-03-03T00:00:00",
                "category": "RACE_B",
            },
        ]
    )
    context = await gather_context(client, ContextPlan.standard(TODAY), "status check", today=TODAY)
    assert [race.name for race in context.goal_races] == ["Coming Race"]


def test_budget_keeps_goal_races_while_other_sections_are_trimmed() -> None:
    race = GoalRace(
        event_id=1,
        name="Autumn Trail Race",
        date=date(2024, 2, 20),
        category="RACE_B",
        days_to_race=19,
        weeks_to_race=3,
    )
    activities = [Activity.model_validate(make_activity("fx-old", 1))]
    wellness = [Wellness.model_validate(make_wellness(28))]
    target = CoachContext(focus="f", goal_races=[race], today=TODAY)
    context = build_within_budget(
        focus="f",
        recent_activities=activities,
        wellness=wellness,
        upcoming_events=[],
        sport_settings=[],
        goal_races=[race],
        user_feedback=None,
        activity_detail=None,
        max_tokens=target.estimated_tokens(),
        today=TODAY,
    )
    assert [item.name for item in context.goal_races] == ["Autumn Trail Race"]
    assert context.recent_activities == []
    assert context.wellness == []


def _goal_race(event_id: int, day: date, *, description: str | None) -> GoalRace:
    return GoalRace(
        event_id=event_id,
        name=f"Race {event_id}",
        date=day,
        category="RACE_B",
        days_to_race=(day - TODAY).days,
        weeks_to_race=1,
        description=description,
    )


def _budget_for(races: list[GoalRace], *, overshoot: int) -> CoachContext:
    base = CoachContext(focus="f", goal_races=races, today=TODAY)
    return build_within_budget(
        focus="f",
        recent_activities=[],
        wellness=[],
        upcoming_events=[],
        sport_settings=[],
        goal_races=races,
        user_feedback=None,
        activity_detail=None,
        max_tokens=base.data_tokens() - overshoot,
        today=TODAY,
    )


def test_budget_trims_the_furthest_goal_race_description_before_dropping_it() -> None:
    near = _goal_race(1, date(2024, 2, 10), description="NEAR " * 200)
    far = _goal_race(2, date(2024, 2, 20), description="FAR " * 200)
    assert far.description is not None
    context = _budget_for([near, far], overshoot=20)
    assert [race.event_id for race in context.goal_races] == [1, 2]
    nearest, furthest = context.goal_races
    assert furthest.description is not None
    assert furthest.description != far.description
    assert far.description.startswith(furthest.description)
    assert furthest.description_truncated is True
    assert nearest.description == near.description
    assert nearest.description_truncated is None
    assert context.data_tokens() <= context.max_tokens


def test_budget_drops_the_furthest_goal_race_when_its_description_cannot_free_enough() -> None:
    near = _goal_race(1, date(2024, 2, 10), description=None)
    far = _goal_race(2, date(2024, 2, 20), description="x")
    context = _budget_for([near, far], overshoot=30)
    assert [race.event_id for race in context.goal_races] == [1]


def test_budget_leaves_a_goal_race_description_untouched_within_budget() -> None:
    race = _goal_race(1, date(2024, 2, 10), description="Full plan " * 100)
    context = _budget_for([race], overshoot=0)
    assert context.goal_races[0].description == race.description
    assert context.goal_races[0].description_truncated is None


def test_budget_terminates_with_many_large_goal_race_descriptions() -> None:
    races = [
        _goal_race(index, date(2024, 2, 10) + timedelta(days=index), description="P" * 2000)
        for index in range(5)
    ]
    base = CoachContext(focus="f", goal_races=races, today=TODAY)
    context = _budget_for(races, overshoot=base.data_tokens() // 2)
    assert context.data_tokens() <= context.max_tokens


async def test_irregular_summary_spacing_degrades_to_no_rollup(settings: Settings) -> None:
    client = make_intervals_client(
        athlete_summary=[make_summary_week("2024-01-22"), make_summary_week("2024-01-24")]
    )
    context = await gather_context(client, ContextPlan.standard(TODAY), "status check", today=TODAY)
    assert context.training_rollup == []


def test_budget_keeps_pinned_activities_under_pressure() -> None:
    activities = [
        Activity.model_validate(make_activity(f"fx-{index}", index)) for index in range(1, 10)
    ]
    pinned = activities[0]
    target = CoachContext(focus="f", recent_activities=[pinned], today=TODAY)
    context = build_within_budget(
        focus="f",
        recent_activities=activities,
        wellness=[],
        upcoming_events=[],
        sport_settings=[],
        activity_keep_ids={pinned.id},
        user_feedback=None,
        activity_detail=None,
        max_tokens=target.estimated_tokens(),
        today=TODAY,
    )
    assert [activity.id for activity in context.recent_activities] == [pinned.id]


def test_budget_trims_splits_from_the_end_under_pressure() -> None:
    splits = [
        ActivitySplit(label=f"km {index}", distance_m=1000.0, time_s=300) for index in range(1, 41)
    ]
    target = CoachContext(focus="f", activity_splits=splits[:10], today=TODAY)
    context = build_within_budget(
        focus="f",
        recent_activities=[],
        wellness=[],
        upcoming_events=[],
        sport_settings=[],
        activity_splits=splits,
        user_feedback=None,
        activity_detail=None,
        max_tokens=target.estimated_tokens(),
        today=TODAY,
    )
    assert [split.label for split in context.activity_splits] == [
        f"km {index}" for index in range(1, 11)
    ]


async def test_budget_keeps_the_activity_detail_over_the_splits() -> None:
    detail = Activity.model_validate(make_activity("fx-a", 5))
    splits = [
        ActivitySplit(label=f"km {index}", distance_m=1000.0, time_s=300) for index in range(1, 41)
    ]
    target = CoachContext(focus="f", activity_detail=detail, today=TODAY)
    context = build_within_budget(
        focus="f",
        recent_activities=[],
        wellness=[],
        upcoming_events=[],
        sport_settings=[],
        activity_detail=detail,
        activity_splits=splits,
        user_feedback=None,
        max_tokens=target.estimated_tokens(),
        today=TODAY,
    )

    assert context.activity_detail is not None
    assert context.activity_splits == []


def test_the_execution_review_is_evicted_after_the_block_plan(settings: Settings) -> None:
    review = ExecutionReview(
        weeks=[
            ReviewWeek(
                week_start=date(2024, 1, 29),
                phase="Base",
                status="done",
                band_low=300.0,
                band_high=400.0,
                executed_load=350.0,
                completed=3,
                adherence=1.0,
            )
        ],
        sessions=[],
    )
    base = build_within_budget(
        focus="review",
        recent_activities=[],
        wellness=[],
        upcoming_events=[],
        sport_settings=[],
        execution_review=review,
        user_feedback=None,
        activity_detail=None,
        today=date(2024, 2, 1),
        max_tokens=100_000,
    )

    context = build_within_budget(
        focus="review",
        recent_activities=[],
        wellness=[],
        upcoming_events=[],
        sport_settings=[],
        execution_review=review,
        user_feedback=None,
        activity_detail=None,
        today=date(2024, 2, 1),
        block_plan=[
            BlockWeek(
                week_start=date(2024, 1, 1) + timedelta(days=7 * index),
                phase="Base",
                week_in_phase=(index % 12) + 1,
                role="Endurance",
                target_tss_low=300.0,
                target_tss_high=400.0,
                status="upcoming",
            )
            for index in range(30)
        ],
        max_tokens=base.data_tokens(),
    )

    assert context.block_plan == []
    assert context.execution_review is not None

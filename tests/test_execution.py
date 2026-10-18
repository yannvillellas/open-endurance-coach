from datetime import date, datetime

from open_endurance_coach.analysis.execution import (
    WeekReview,
    aggregate_weeks,
    build_review,
    compare_trajectory,
    match_execution,
)
from open_endurance_coach.schemas.context import BlockWeek, TrainingWeek
from open_endurance_coach.schemas.intervals import Activity, Event

TODAY = date(2024, 2, 12)


def activity(
    aid: str = "a1",
    day: date = date(2024, 2, 10),
    *,
    activity_type: str = "Ride",
    load: float | None = 100.0,
    moving_time: int | None = 3600,
    distance: float | None = 30000.0,
    compliance: float | None = None,
    paired_event_id: int | None = None,
) -> Activity:
    return Activity(
        id=aid,
        start_date_local=datetime(day.year, day.month, day.day, 8, 0),
        type=activity_type,
        name="Synthetic Workout",
        icu_training_load=load,
        moving_time=moving_time,
        distance=distance,
        compliance=compliance,
        paired_event_id=paired_event_id,
    )


def planned(
    event_id: int = 1,
    day: date = date(2024, 2, 10),
    *,
    category: str = "WORKOUT",
    event_type: str | None = "Ride",
    load: float | None = 100.0,
    moving_time: int | None = 3600,
    distance: float | None = 30000.0,
) -> Event:
    return Event(
        id=event_id,
        name="Planned Session",
        start_date_local=datetime(day.year, day.month, day.day, 0, 0),
        category=category,
        type=event_type,
        icu_training_load=load,
        moving_time=moving_time,
        distance=distance,
    )


def test_the_activity_model_parses_the_hub_compliance_and_pairing() -> None:
    parsed = Activity.model_validate(
        {
            "id": "a1",
            "start_date_local": "2024-02-10T08:00:00",
            "type": "Ride",
            "name": "Synthetic",
            "compliance": 92.5,
            "paired_event_id": 7,
        }
    )

    assert parsed.compliance == 92.5
    assert parsed.paired_event_id == 7


def test_the_hub_pairing_matches_even_far_from_the_planned_date() -> None:
    events = [planned(7, date(2024, 2, 9))]
    activities = [activity("a1", date(2024, 2, 11), paired_event_id=7)]

    result = match_execution(events, activities, today=TODAY)

    assert [session.planned.id for session in result.sessions] == [7]
    assert result.sessions[0].activity is not None
    assert result.sessions[0].activity.id == "a1"
    assert result.sessions[0].status == "completed"
    assert result.unplanned == ()


def test_the_hub_pairing_beats_the_date_fallback() -> None:
    events = [planned(1, date(2024, 2, 10))]
    activities = [
        activity("a-near", date(2024, 2, 10)),
        activity("a-paired", date(2024, 2, 12), paired_event_id=1),
    ]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].activity is not None
    assert result.sessions[0].activity.id == "a-paired"
    assert [unplanned.id for unplanned in result.unplanned] == ["a-near"]


def test_the_nearest_compatible_activity_within_one_day_matches() -> None:
    events = [planned(1, date(2024, 2, 10))]
    activities = [activity("a1", date(2024, 2, 9))]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].activity is not None
    assert result.sessions[0].activity.id == "a1"
    assert result.sessions[0].status == "completed"


def test_a_date_two_days_away_is_missed_and_unplanned() -> None:
    events = [planned(1, date(2024, 2, 8))]
    activities = [activity("a1", date(2024, 2, 10))]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].status == "missed"
    assert result.sessions[0].activity is None
    assert [unplanned.id for unplanned in result.unplanned] == ["a1"]


def test_type_families_match_but_other_sports_do_not() -> None:
    events = [planned(1, date(2024, 2, 10), event_type="Ride")]
    activities = [activity("a1", date(2024, 2, 10), activity_type="VirtualRide")]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].activity is not None

    other = match_execution(
        events,
        [activity("a2", date(2024, 2, 10), activity_type="Run")],
        today=TODAY,
    )

    assert other.sessions[0].status == "missed"
    assert [unplanned.id for unplanned in other.unplanned] == ["a2"]


def test_a_planned_session_without_a_type_matches_any_sport() -> None:
    events = [planned(1, date(2024, 2, 10), event_type=None)]
    activities = [activity("a1", date(2024, 2, 10), activity_type="Swim")]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].activity is not None


def test_the_nearest_activity_wins_when_two_are_in_range() -> None:
    events = [planned(1, date(2024, 2, 10))]
    activities = [
        activity("a-late", date(2024, 2, 11)),
        activity("a-same", date(2024, 2, 10)),
    ]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].activity is not None
    assert result.sessions[0].activity.id == "a-same"
    assert [unplanned.id for unplanned in result.unplanned] == ["a-late"]


def test_each_activity_matches_at_most_once() -> None:
    events = [planned(1, date(2024, 2, 10)), planned(2, date(2024, 2, 11))]
    activities = [activity("a1", date(2024, 2, 10))]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].activity is not None
    assert result.sessions[0].activity.id == "a1"
    assert result.sessions[1].status == "missed"
    assert result.unplanned == ()


def test_today_and_the_future_sessions_are_pending_not_missed() -> None:
    events = [
        planned(1, date(2024, 2, 10)),
        planned(2, TODAY),
        planned(3, date(2024, 2, 13)),
    ]
    activities = [activity("a1", date(2024, 2, 10))]

    result = match_execution(events, activities, today=TODAY)

    assert [session.planned.id for session in result.sessions] == [1]
    assert result.sessions[0].status == "completed"
    assert [event.id for event in result.pending] == [2, 3]
    assert result.unplanned == ()


def test_today_is_judged_when_the_session_is_already_done() -> None:
    events = [planned(1, TODAY)]
    activities = [activity("a-today", TODAY, compliance=85.0, paired_event_id=1)]

    result = match_execution(events, activities, today=TODAY)

    assert [session.planned.id for session in result.sessions] == [1]
    assert result.sessions[0].status == "completed"
    assert result.sessions[0].completion == 0.85
    assert result.pending == ()
    assert result.unplanned == ()


def test_a_hub_pair_to_a_future_session_is_judged() -> None:
    events = [planned(1, date(2024, 2, 11)), planned(5, date(2024, 2, 13))]
    activities = [activity("a1", TODAY, paired_event_id=5)]

    result = match_execution(events, activities, today=TODAY)

    assert [session.planned.id for session in result.sessions] == [1, 5]
    assert result.sessions[0].status == "missed"
    assert result.sessions[1].status == "completed"
    assert result.pending == ()


def test_a_today_session_with_a_small_start_reads_partial_not_missed() -> None:
    events = [planned(1, TODAY, load=100.0)]
    activities = [activity("a1", TODAY, load=15.0)]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].status == "partial"
    assert result.sessions[0].completion == 0.15


def test_the_same_numbers_judge_normally_once_the_day_is_over() -> None:
    events = [planned(1, date(2024, 2, 11), load=100.0)]
    activities = [activity("a1", date(2024, 2, 11), load=15.0)]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].status == "missed"


def test_an_activity_paired_outside_the_review_is_not_reused() -> None:
    events = [planned(1, date(2024, 2, 10))]
    activities = [activity("a1", date(2024, 2, 10), paired_event_id=999)]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].status == "missed"
    assert [unplanned.id for unplanned in result.unplanned] == ["a1"]


def test_a_zero_compliance_is_read_as_zero() -> None:
    events = [planned(1, date(2024, 2, 10), load=100.0)]
    activities = [activity("a1", date(2024, 2, 10), load=100.0, compliance=0.0, paired_event_id=1)]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].status == "missed"
    assert result.sessions[0].completion == 0.0


def test_an_unmatched_activity_dated_today_is_not_unplanned() -> None:
    events = [planned(1, date(2024, 2, 10))]
    activities = [activity("a1", date(2024, 2, 10)), activity("a-today", TODAY)]

    result = match_execution(events, activities, today=TODAY)

    assert [session.planned.id for session in result.sessions] == [1]
    assert result.unplanned == ()


def test_compliance_classifies_first_even_when_the_load_matches() -> None:
    events = [planned(1, load=100.0)]
    activities = [activity("a1", load=100.0, compliance=60.0, paired_event_id=1)]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].status == "partial"
    assert result.sessions[0].completion == 0.6


def test_the_load_ratio_classifies_without_compliance() -> None:
    events = [planned(1, load=100.0)]

    completed = match_execution(events, [activity("a1", load=80.0)], today=TODAY)
    partial = match_execution(events, [activity("a1", load=50.0)], today=TODAY)
    missed = match_execution(events, [activity("a1", load=10.0)], today=TODAY)

    assert completed.sessions[0].status == "completed"
    assert partial.sessions[0].status == "partial"
    assert missed.sessions[0].status == "missed"


def test_duration_then_distance_are_the_fallbacks() -> None:
    events = [planned(1, load=None, moving_time=3600, distance=None)]

    by_time = match_execution(
        events,
        [activity("a1", load=None, moving_time=1800, distance=None)],
        today=TODAY,
    )
    by_distance = match_execution(
        [planned(1, load=None, moving_time=None, distance=30000.0)],
        [activity("a1", load=None, moving_time=None, distance=15000.0)],
        today=TODAY,
    )

    assert by_time.sessions[0].status == "partial"
    assert by_time.sessions[0].completion == 0.5
    assert by_distance.sessions[0].status == "partial"
    assert by_distance.sessions[0].completion == 0.5


def test_a_matched_session_without_comparable_numbers_is_completed() -> None:
    events = [planned(1, load=None, moving_time=None, distance=None)]
    activities = [activity("a1", load=None, moving_time=None, distance=None)]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].status == "completed"
    assert result.sessions[0].completion is None


def test_race_events_count_as_planned() -> None:
    events = [planned(1, date(2024, 2, 10), category="RACE_A", load=300.0)]
    activities = [activity("a1", date(2024, 2, 10), load=290.0)]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].status == "completed"
    assert result.unplanned == ()


def test_non_planned_events_are_ignored() -> None:
    events = [planned(1, date(2024, 2, 10), category="NOTE")]
    activities = [activity("a1", date(2024, 2, 10))]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions == ()
    assert result.pending == ()
    assert [unplanned.id for unplanned in result.unplanned] == ["a1"]


def test_the_result_is_ordered_and_deterministic() -> None:
    events = [planned(2, date(2024, 2, 11)), planned(1, date(2024, 2, 9))]
    activities = [activity("a-late", date(2024, 2, 11)), activity("a-early", date(2024, 2, 9))]

    first = match_execution(events, activities, today=TODAY)
    second = match_execution(events, activities, today=TODAY)

    assert first == second
    assert [session.planned.id for session in first.sessions] == [1, 2]
    assert first.unplanned == ()


def test_duplicate_pairing_is_deterministic() -> None:
    events = [planned(7, date(2024, 2, 10))]
    activities = [
        activity("a9", date(2024, 2, 10), paired_event_id=7),
        activity("a1", date(2024, 2, 10), paired_event_id=7),
    ]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].activity is not None
    assert result.sessions[0].activity.id == "a1"
    assert [unplanned.id for unplanned in result.unplanned] == ["a9"]


def test_weeks_fold_sessions_loads_and_adherence() -> None:
    events = [
        planned(1, date(2024, 2, 5), load=100.0),
        planned(2, date(2024, 2, 6), load=100.0),
        planned(3, date(2024, 2, 7), load=100.0),
        planned(4, TODAY, load=80.0),
    ]
    activities = [
        activity("a1", date(2024, 2, 5), load=100.0),
        activity("a2", date(2024, 2, 6), load=50.0),
        activity("a4", TODAY, load=80.0),
    ]
    match = match_execution(events, activities, today=TODAY)

    reviews = {review.week_start: review for review in aggregate_weeks(match, today=TODAY)}

    past = reviews[date(2024, 2, 5)]
    assert past.planned_load == 300.0
    assert past.executed_load == 150.0
    assert (past.completed, past.partial, past.missed) == (1, 1, 1)
    assert past.adherence == 1 / 3
    assert past.current is False

    current = reviews[date(2024, 2, 12)]
    assert current.planned_load == 80.0
    assert current.executed_load == 80.0
    assert (current.completed, current.partial, current.missed) == (1, 0, 0)
    assert current.adherence == 1.0
    assert current.current is True


def test_a_gap_week_shows_its_empty_row() -> None:
    events = [planned(1, date(2024, 1, 29), load=100.0), planned(2, TODAY, load=80.0)]
    activities = [activity("a1", date(2024, 1, 29), load=100.0), activity("a2", TODAY, load=80.0)]
    match = match_execution(events, activities, today=TODAY)

    reviews = aggregate_weeks(match, today=TODAY)

    assert [review.week_start for review in reviews] == [
        date(2024, 1, 29),
        date(2024, 2, 5),
        date(2024, 2, 12),
    ]
    gap = reviews[1]
    assert gap.planned_load is None
    assert gap.executed_load is None
    assert (gap.completed, gap.partial, gap.missed) == (0, 0, 0)
    assert gap.adherence is None
    assert gap.current is False


def test_unplanned_load_and_share() -> None:
    events = [planned(1, date(2024, 2, 5), load=100.0)]
    activities = [
        activity("a1", date(2024, 2, 5), load=100.0),
        activity("a2", date(2024, 2, 6), activity_type="Run", load=50.0),
    ]
    match = match_execution(events, activities, today=TODAY)

    reviews = {review.week_start: review for review in aggregate_weeks(match, today=TODAY)}

    week = reviews[date(2024, 2, 5)]
    assert week.unplanned_load == 50.0
    assert week.unplanned_share == 1 / 3


def test_pending_sessions_are_not_folded() -> None:
    events = [
        planned(1, date(2024, 2, 5), load=100.0),
        planned(2, TODAY),
        planned(3, date(2024, 2, 20)),
    ]
    activities = [activity("a1", date(2024, 2, 5), load=100.0)]
    match = match_execution(events, activities, today=TODAY)

    reviews = aggregate_weeks(match, today=TODAY)

    assert [review.week_start for review in reviews] == [date(2024, 2, 5), date(2024, 2, 12)]
    current = reviews[-1]
    assert current.current is True
    assert current.planned_load is None
    assert (current.completed, current.partial, current.missed) == (0, 0, 0)
    assert current.adherence is None


def test_only_a_future_judged_session_spans_that_future_week() -> None:
    events = [planned(5, date(2024, 2, 20), load=100.0)]
    activities = [activity("a1", TODAY, paired_event_id=5, load=100.0)]
    match = match_execution(events, activities, today=TODAY)

    reviews = aggregate_weeks(match, today=TODAY)

    assert [review.week_start for review in reviews] == [date(2024, 2, 19)]
    assert reviews[0].current is False


def test_a_fully_missed_week_reads_zero_adherence() -> None:
    events = [planned(1, date(2024, 2, 5), load=100.0)]
    match = match_execution(events, [], today=TODAY)

    reviews = {review.week_start: review for review in aggregate_weeks(match, today=TODAY)}

    week = reviews[date(2024, 2, 5)]
    assert week.missed == 1
    assert week.adherence == 0.0


def test_an_unplanned_only_week_has_no_adherence_and_a_full_share() -> None:
    activities = [activity("a2", date(2024, 2, 8), activity_type="Run", load=50.0)]
    match = match_execution([], activities, today=TODAY)

    reviews = {review.week_start: review for review in aggregate_weeks(match, today=TODAY)}

    week = reviews[date(2024, 2, 5)]
    assert week.adherence is None
    assert week.unplanned_load == 50.0
    assert week.unplanned_share == 1.0


def weekly_review(
    week_start: date,
    *,
    planned: float | None = None,
    executed: float | None = None,
    completed: int = 0,
    partial: int = 0,
    missed: int = 0,
    unplanned: float | None = None,
    adherence: float | None = None,
    share: float | None = None,
    current: bool = False,
) -> WeekReview:
    return WeekReview(
        week_start=week_start,
        planned_load=planned,
        executed_load=executed,
        completed=completed,
        partial=partial,
        missed=missed,
        adherence=adherence,
        unplanned_load=unplanned,
        unplanned_share=share,
        current=current,
    )


def block_row(
    week_start: date,
    *,
    phase: str = "Base",
    status: str = "done",
    low: float | None = 300.0,
    high: float | None = 400.0,
) -> BlockWeek:
    return BlockWeek(
        week_start=week_start,
        phase=phase,
        week_in_phase=1,
        role="Endurance",
        target_tss_low=low,
        target_tss_high=high,
        status=status,
    )


def rollup_week(
    week_start: date, *, fitness: float | None = None, ramp: float | None = None
) -> TrainingWeek:
    return TrainingWeek(
        week_start=week_start, sessions=4, time_s=7200, load=300.0, fitness=fitness, ramp_rate=ramp
    )


def test_efficacy_lines_up_the_adjusted_week_the_status_and_the_hub_movement() -> None:
    reviews = [
        weekly_review(date(2024, 1, 29), executed=250.0, completed=1),  # trained in a rest week
        weekly_review(date(2024, 2, 5), planned=320.0, executed=300.0, completed=3, unplanned=50.0),
        weekly_review(date(2024, 2, 12), executed=120.0, completed=1, current=True),
        weekly_review(date(2024, 1, 22), executed=90.0, completed=1),  # before the block: skipped
    ]
    block = [
        block_row(date(2024, 1, 29), status="rest", low=0.0, high=0.0),
        block_row(date(2024, 2, 5), status="done", low=300.0, high=400.0),
        block_row(date(2024, 2, 12), status="under", low=200.0, high=300.0),
    ]
    rollup = [
        rollup_week(date(2024, 1, 29), fitness=58.0, ramp=0.5),
        rollup_week(date(2024, 2, 5), fitness=60.0, ramp=0.7),
        rollup_week(date(2024, 2, 12), fitness=62.0, ramp=0.4),
    ]

    rows = compare_trajectory(reviews, block, rollup)

    assert [row.week_start for row in rows] == [
        date(2024, 1, 29),
        date(2024, 2, 5),
        date(2024, 2, 12),
    ]
    rest_week = rows[0]
    assert rest_week.status == "rest"  # the plan status is read as-is, never re-derived
    assert rest_week.executed_load == 250.0
    first = rows[1]
    assert (first.phase, first.status) == ("Base", "done")
    assert (first.band_low, first.band_high) == (300.0, 400.0)
    assert first.executed_load == 350.0
    assert (first.fitness, first.ramp) == (60.0, 0.7)
    assert rows[2].executed_load == 120.0
    assert (rows[2].phase, rows[2].status) == ("Base", "under")


def test_efficacy_leaves_missing_values_empty() -> None:
    reviews = [weekly_review(date(2024, 2, 5), planned=None, executed=None)]
    block = [block_row(date(2024, 2, 5), status="rest", low=0.0, high=0.0)]

    rows = compare_trajectory(reviews, block, [])

    assert len(rows) == 1
    assert rows[0].executed_load is None
    assert rows[0].fitness is None
    assert rows[0].ramp is None
    assert (rows[0].band_low, rows[0].band_high) == (0.0, 0.0)


def test_efficacy_keeps_the_review_order_and_reads_zero_loads() -> None:
    reviews = [
        weekly_review(date(2024, 2, 12), executed=0.0),
        weekly_review(date(2024, 2, 5), unplanned=50.0),
    ]
    block = [
        block_row(date(2024, 2, 5), status="rest", low=0.0, high=0.0),
        block_row(date(2024, 2, 12), status="rest", low=0.0, high=0.0),
    ]
    rollup = [
        rollup_week(date(2024, 2, 5), fitness=0.0, ramp=0.0),
        rollup_week(date(2024, 2, 12)),
    ]

    rows = compare_trajectory(reviews, block, rollup)

    assert [row.week_start for row in rows] == [date(2024, 2, 12), date(2024, 2, 5)]
    assert rows[0].executed_load == 0.0
    assert rows[1].executed_load == 50.0
    assert (rows[1].fitness, rows[1].ramp) == (0.0, 0.0)
    assert rows[0].fitness is None


def test_build_review_keeps_the_week_rows_and_the_session_detail() -> None:
    events = [
        planned(1, date(2024, 2, 5), load=100.0),
        planned(2, TODAY, load=80.0),
    ]
    activities = [
        activity("a1", date(2024, 2, 5), load=90.0, compliance=95.0, paired_event_id=1),
        activity("a2", date(2024, 2, 7), activity_type="Run", load=40.0),
        activity("a3", TODAY, load=80.0, compliance=0.0),
    ]
    match = match_execution(events, activities, today=TODAY)
    weeks = aggregate_weeks(match, today=TODAY)
    block = [
        block_row(date(2024, 2, 5), status="done", low=90.0, high=110.0),
        block_row(TODAY, status="current", low=60.0, high=90.0),
    ]
    rollup = [
        rollup_week(date(2024, 2, 5), fitness=52.0, ramp=3.2),
        rollup_week(TODAY, fitness=54.0, ramp=2.0),
    ]
    efficacy = compare_trajectory(weeks, block, rollup)

    review = build_review(match, weeks, efficacy)

    assert [week.status for week in review.weeks] == ["done", "current"]
    assert review.weeks[0].band_low == 90.0
    assert review.weeks[0].completed == 1
    assert review.weeks[0].planned_load == 100.0
    assert review.weeks[0].executed_load == 130.0  # matched 90 + unplanned 40
    assert review.weeks[0].fitness == 52.0
    assert [(session.date, session.status) for session in review.sessions] == [
        (date(2024, 2, 5), "completed"),
        (date(2024, 2, 7), "unplanned"),
        (TODAY, "completed"),
    ]
    first = review.sessions[0]
    assert (first.planned, first.planned_load) == ("Planned Session", 100.0)
    assert (first.actual, first.actual_load) == ("Synthetic Workout", 90.0)
    assert first.compliance == 95.0
    assert review.sessions[-1].compliance is None  # unpaired sentinel never leaks


def test_an_unscored_fallback_match_reads_the_load_ratio() -> None:
    events = [planned(1, date(2024, 2, 10), load=100.0)]
    activities = [activity("a1", date(2024, 2, 10), load=100.0, compliance=0.0)]

    result = match_execution(events, activities, today=TODAY)

    assert result.sessions[0].paired is False
    assert result.sessions[0].status == "completed"
    assert result.sessions[0].completion == 1.0


def test_weeks_follow_the_block_cadence() -> None:
    events = [planned(1, date(2024, 2, 5), load=100.0)]
    activities = [activity("a1", date(2024, 2, 5), load=100.0)]
    match = match_execution(events, activities, today=TODAY)

    reviews = aggregate_weeks(match, today=TODAY, origin=date(2024, 2, 4))

    assert [review.week_start for review in reviews] == [
        date(2024, 2, 4),
        date(2024, 2, 11),
    ]
    assert reviews[1].current is True


def test_a_done_early_session_is_counted_in_its_week() -> None:
    events = [
        planned(1, date(2024, 2, 5), load=100.0),
        planned(5, date(2024, 2, 20), load=100.0),
    ]
    activities = [
        activity("a1", date(2024, 2, 5), load=100.0),
        activity("a2", TODAY, paired_event_id=5, load=100.0),
    ]
    match = match_execution(events, activities, today=TODAY)

    reviews = aggregate_weeks(match, today=TODAY)

    assert [review.week_start for review in reviews] == [
        date(2024, 2, 5),
        date(2024, 2, 12),
        date(2024, 2, 19),
    ]
    assert reviews[-1].completed == 1
    assert reviews[-1].executed_load == 100.0

from datetime import date, datetime

from open_endurance_coach.analysis.execution import match_execution
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
    activities = [activity("a-today", TODAY, compliance=85.0)]

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
    activities = [activity("a1", date(2024, 2, 10), load=100.0, compliance=0.0)]

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
    activities = [activity("a1", load=100.0, compliance=60.0)]

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

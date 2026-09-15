from datetime import date, timedelta

import pytest

from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.extractors.deep import DeepHistoricalExtractor, detect_deep_query
from open_endurance_coach.extractors.standard import StandardExtractor, macro_phase, training_rollup
from open_endurance_coach.schemas.context import CoachContext, TrainingWeek
from open_endurance_coach.schemas.intervals import Activity, Wellness

from .fakes import make_activity, make_intervals_client, make_summary_week, make_wellness

TODAY = date(2024, 2, 1)


async def test_standard_extraction_populates_all_sections(settings: Settings) -> None:
    extractor = StandardExtractor(settings, make_intervals_client())
    context = await extractor.extract("Analyze this week", today=TODAY)
    assert len(context.recent_activities) == 5
    assert len(context.wellness) == 3
    assert len(context.upcoming_events) == 2
    assert context.sport_settings[0].ftp == 250.0
    assert context.goal_races == []
    assert context.activity_detail is None
    assert context.user_feedback is None
    assert context.today == TODAY
    assert [week.week_start for week in context.training_rollup] == [
        date(2024, 1, 23),
        date(2024, 1, 30),
    ]
    assert context.training_rollup[-1].partial is True
    assert context.training_rollup[-1].sessions == 2


async def test_standard_extraction_uses_expected_windows(settings: Settings) -> None:
    client = make_intervals_client()
    extractor = StandardExtractor(settings, client)
    await extractor.extract("status check", user_feedback="felt tired", today=TODAY)
    assert client.calls == [
        ("activities", "2024-01-18", "2024-02-02"),
        ("wellness", "2024-01-25", "2024-02-02"),
        ("events", "2024-02-01", "2024-02-15", None),
        ("events", "2024-02-01", "2024-05-31", "RACE_A,RACE_B,RACE_C"),
        ("athlete_summary", "2023-11-03", "2024-02-01"),
        ("sport_settings",),
    ]


async def test_standard_extraction_keeps_newest_first(settings: Settings) -> None:
    extractor = StandardExtractor(settings, make_intervals_client())
    context = await extractor.extract("status check", today=TODAY)
    assert [item.id for item in context.recent_activities] == [
        "fx-a",
        "fx-b",
        "fx-c",
        "fx-d",
        "fx-e",
    ]


async def test_budget_overrun_trims_oldest_activities_first(settings: Settings) -> None:
    extractor = StandardExtractor(settings, make_intervals_client())
    context = await extractor.extract("status check", today=TODAY, max_tokens=250)
    assert context.estimated_tokens() <= 250
    assert len(context.recent_activities) < 5
    assert [item.id for item in context.recent_activities] == [
        "fx-a",
        "fx-b",
        "fx-c",
        "fx-d",
    ][: len(context.recent_activities)]
    assert len(context.wellness) == 3
    assert len(context.upcoming_events) == 2


async def test_budget_too_small_to_fit_focus_raises(settings: Settings) -> None:
    extractor = StandardExtractor(settings, make_intervals_client())
    with pytest.raises(RuntimeError, match="token budget"):
        await extractor.extract("status check", today=TODAY, max_tokens=1)


async def test_standard_extraction_builds_goal_races(settings: Settings) -> None:
    client = make_intervals_client(
        events=[
            {
                "id": 90001,
                "name": "Spring Half",
                "start_date_local": "2024-03-01T00:00:00",
                "category": "RACE_A",
                "type": "Run",
            },
            {
                "id": 90002,
                "name": "Club Crit",
                "start_date_local": "2024-02-08T00:00:00",
                "category": "RACE_B",
                "type": "Ride",
            },
            {
                "id": 90003,
                "name": "Tempo Session",
                "start_date_local": "2024-02-05T00:00:00",
                "category": "WORKOUT",
            },
        ]
    )
    extractor = StandardExtractor(settings, client)
    context = await extractor.extract("status check", today=TODAY)
    assert [race.category for race in context.goal_races] == ["RACE_B", "RACE_A"]
    nearest, furthest = context.goal_races
    assert nearest.event_id == 90002
    assert nearest.name == "Club Crit"
    assert nearest.date == date(2024, 2, 8)
    assert nearest.type == "Ride"
    assert nearest.days_to_race == 7
    assert nearest.weeks_to_race == 1
    assert nearest.phase == "Taper"
    assert furthest.days_to_race == 29
    assert furthest.weeks_to_race == 5
    assert furthest.phase == "Build"


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (0, "Race week"),
        (7, "Taper"),
        (8, "Peak"),
        (28, "Peak"),
        (29, "Build"),
        (84, "Build"),
        (85, "Base"),
        (120, "Base"),
    ],
)
def test_macro_phase_maps_days_to_race(days: int, expected: str) -> None:
    assert macro_phase(days) == expected


async def test_budget_keeps_recent_activities_and_drops_wellness_first(
    settings: Settings,
) -> None:
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
        goal_races=[],
        user_feedback=None,
        activity_detail=None,
        max_tokens=probe.estimated_tokens(),
        today=TODAY,
    )
    assert len(context.recent_activities) == 2
    assert context.wellness == []


@pytest.mark.parametrize(
    ("focus", "lookback", "metric"),
    [
        ("trend in heart rate over the last 6 weeks", 42, "heart_rate"),
        ("how did my power improve on hills in the last 3 months", 90, "elevation"),
        ("progress on climbs over the last 4 weeks", 28, "elevation"),
        ("heart rate evolution", 90, "heart_rate"),
        ("heart rate improve on hilly sections", 90, "heart_rate"),
    ],
)
def test_detect_deep_query_parses_focus(focus: str, lookback: int, metric: str) -> None:
    query = detect_deep_query(focus)
    assert query is not None
    assert query.lookback_days == lookback
    assert query.metric_focus == metric


def test_hill_query_without_a_sport_keeps_every_type() -> None:
    query = detect_deep_query("heart rate improve on hilly sections")
    assert query is not None
    assert query.metric_focus == "heart_rate"
    assert query.activity_types == frozenset()


def test_hill_query_naming_a_ride_keeps_the_cycling_family() -> None:
    query = detect_deep_query("heart rate improve on hilly bike sections")
    assert query is not None
    assert {"Ride", "VirtualRide", "GravelRide", "MountainBikeRide"} <= query.activity_types


def test_zwift_query_keeps_virtual_rides() -> None:
    query = detect_deep_query("heart rate trend on my zwift rides")
    assert query is not None
    assert "VirtualRide" in query.activity_types


def test_detect_deep_query_on_past_activity_reference() -> None:
    query = detect_deep_query("check the activity on 2023-01-15", today=TODAY)
    assert query is not None
    assert query.lookback_days >= 382


def test_detect_deep_query_on_worded_past_reference() -> None:
    assert detect_deep_query("my race one year ago", today=TODAY) is not None
    assert detect_deep_query("28th of september 2023 race", today=TODAY) is not None


def test_detect_deep_query_ignores_impossible_dates() -> None:
    assert detect_deep_query("race on 2026-13-45", today=TODAY) is None
    assert detect_deep_query("31st of april 2023 race", today=TODAY) is None


def test_detect_deep_query_ignores_recent_references() -> None:
    assert detect_deep_query("how was my run on 2024-01-30?", today=TODAY) is None


def test_detect_deep_query_returns_none_for_plain_focus() -> None:
    assert detect_deep_query("status check") is None
    assert detect_deep_query("Analyze this week") is None


def test_climb_focus_without_a_sport_keeps_every_type() -> None:
    query = detect_deep_query("progress on climbs over the last 4 weeks")
    assert query is not None
    assert query.activity_types == frozenset()


async def test_deep_extraction_fetches_filtered_window_and_detail(settings: Settings) -> None:
    client = make_intervals_client(
        detail={
            "start_date_local": "2024-01-20T08:00:00",
            "type": "Ride",
            "name": "Synthetic Workout",
            "icu_intervals": [{"average_heartrate": 165}],
        }
    )
    focus = "how did my heart rate improve on hills on my bike in the last 3 months"
    extractor = DeepHistoricalExtractor(settings, client)
    context = await extractor.extract(focus, query=detect_deep_query(focus), today=TODAY)
    activities_oldest = next(call for call in client.calls if call[0] == "activities")[1]
    assert activities_oldest == "2023-11-03"
    assert context.activity_detail is not None
    assert context.activity_detail.id == "fx-a"
    assert context.activity_detail.icu_intervals is not None
    assert len(context.recent_activities) == 4


async def test_deep_extraction_keeps_every_sport_when_none_is_named(
    settings: Settings,
) -> None:
    client = make_intervals_client()
    focus = "how did my heart rate improve on hills in the last 3 months"
    extractor = DeepHistoricalExtractor(settings, client)
    context = await extractor.extract(focus, query=detect_deep_query(focus), today=TODAY)
    assert {activity.type for activity in context.recent_activities} == {"Ride", "Run"}


async def test_deep_extraction_rejects_non_deep_focus(settings: Settings) -> None:
    extractor = DeepHistoricalExtractor(settings, make_intervals_client())
    with pytest.raises(ValueError, match="deep query"):
        await extractor.extract("status check", query=None, today=TODAY)


async def test_deep_extraction_respects_budget(settings: Settings) -> None:
    extractor = DeepHistoricalExtractor(settings, make_intervals_client())
    context = await extractor.extract(
        "heart rate evolution",
        query=detect_deep_query("heart rate evolution"),
        today=TODAY,
        max_tokens=150,
    )
    assert isinstance(context, CoachContext)
    assert context.estimated_tokens() <= 150


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


async def test_deep_extraction_carries_goal_races_and_rollup(settings: Settings) -> None:
    client = make_intervals_client(
        events=[
            {
                "name": "Trail Race",
                "start_date_local": "2024-02-20T00:00:00",
                "category": "RACE_B",
                "type": "Run",
            }
        ]
    )
    focus = "how much did my heart rate improve over the last 3 months"
    extractor = DeepHistoricalExtractor(settings, client)
    context = await extractor.extract(focus, query=detect_deep_query(focus), today=TODAY)
    assert [race.name for race in context.goal_races] == ["Trail Race"]
    assert len(context.training_rollup) == 2
    race_call = next(call for call in client.calls if call[0] == "events" and call[3] is not None)
    assert race_call[1:] == ("2024-02-01", "2024-05-31", "RACE_A,RACE_B,RACE_C")
    summary_call = next(call for call in client.calls if call[0] == "athlete_summary")
    assert summary_call[1:] == ("2023-11-03", "2024-02-01")

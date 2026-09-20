from datetime import date, timedelta
from typing import Any

import pytest

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.extractors.deep import DeepHistoricalExtractor, detect_deep_query
from open_endurance_coach.extractors.standard import StandardExtractor, macro_phase, training_rollup
from open_endurance_coach.schemas.context import CoachContext, GoalRace, TrainingWeek
from open_endurance_coach.schemas.intervals import Activity, ActivitySplit, Event, Wellness

from .fakes import (
    TODAY,
    make_activity,
    make_event,
    make_intervals_client,
    make_summary_week,
    make_wellness,
)


async def test_standard_extraction_populates_all_sections(settings: Settings) -> None:
    extractor = StandardExtractor(settings, make_intervals_client())
    context = await extractor.extract("Analyze this week", today=TODAY)
    assert len(context.recent_activities) == 5
    assert len(context.wellness) == 3
    assert len(context.upcoming_events) == 2
    assert context.sport_settings[0].ftp == 250.0
    assert context.activity_detail is None
    assert context.user_feedback is None
    assert context.today == TODAY


async def test_standard_extraction_uses_expected_windows(settings: Settings) -> None:
    client = make_intervals_client()
    extractor = StandardExtractor(settings, client)
    await extractor.extract("status check", user_feedback="felt tired", today=TODAY)
    assert client.calls == [
        ("activities", "2024-01-18", "2024-02-02"),
        ("wellness", "2024-01-25", "2024-02-02"),
        ("events", "2024-01-18", "2024-02-15", None),
        ("events", "2024-02-01", "2024-05-31", "RACE_A,RACE_B,RACE_C"),
        ("athlete_summary", "2023-11-03", "2024-02-01"),
        ("sport_settings",),
    ]


async def test_standard_extraction_splits_past_and_upcoming_events(settings: Settings) -> None:
    client = make_intervals_client(
        events=[
            make_event(0, "2024-01-29", name="Older past session"),
            make_event(1, "2024-01-30", name="Prescribed hill session"),
            make_event(2, "2024-02-01", name="Today session"),
            make_event(3, "2024-02-03", name="Tempo Session"),
        ]
    )
    extractor = StandardExtractor(settings, client)
    context = await extractor.extract("review yesterday", today=TODAY)
    assert [event.name for event in context.recent_events] == [
        "Prescribed hill session",
        "Older past session",
    ]
    assert [event.name for event in context.upcoming_events] == [
        "Today session",
        "Tempo Session",
    ]


async def test_deep_extraction_splits_past_and_upcoming_events(settings: Settings) -> None:
    client = make_intervals_client(
        events=[
            make_event(1, "2024-01-30", name="Past session"),
            make_event(2, "2024-02-10", name="Future session"),
        ]
    )
    focus = "how did my heart rate improve on hills in the last 3 months"
    extractor = DeepHistoricalExtractor(settings, client)
    context = await extractor.extract(focus, query=detect_deep_query(focus), today=TODAY)
    assert ("events", "2024-01-18", "2024-02-15", None) in client.calls
    assert [event.name for event in context.recent_events] == ["Past session"]
    assert [event.name for event in context.upcoming_events] == ["Future session"]


async def test_budget_drops_past_events_before_upcoming(settings: Settings) -> None:
    events = [
        make_event(
            100 + index, (TODAY - timedelta(days=index + 1)).isoformat(), name=f"Past {index}"
        )
        for index in range(14)
    ] + [make_event(200, (TODAY + timedelta(days=1)).isoformat(), name="Tomorrow")]
    extractor = StandardExtractor(settings, make_intervals_client(events=events))
    context = await extractor.extract("status check", today=TODAY, max_tokens=500)
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


async def test_deep_extraction_includes_activity_splits(settings: Settings) -> None:
    streams = {
        "time": list(range(601)),
        "distance": [index * 1000 / 300 for index in range(601)],
        "heartrate": [150] * 601,
    }
    client = make_intervals_client(streams=streams)
    focus = "how did my heart rate improve on hills in the last 3 months"
    extractor = DeepHistoricalExtractor(settings, client)
    context = await extractor.extract(focus, query=detect_deep_query(focus), today=TODAY)

    assert [split.label for split in context.activity_splits] == ["km 1", "km 2"]
    assert context.activity_splits[0].average_speed_kmh == 12.0
    assert context.activity_splits[0].pace_s_per_km is None
    assert context.activity_splits[0].average_heartrate == 150
    assert any(call[0] == "streams" and "watts" in call[2] for call in client.calls)


async def test_deep_extraction_splits_follow_the_stream_extent(settings: Settings) -> None:
    streams: dict[str, list[Any]] = {
        "time": list(range(11401)),
        "distance": [index * 1000 / 300 for index in range(11401)],
    }
    client = make_intervals_client(
        streams=streams,
        detail={
            "start_date_local": "2024-01-20T08:00:00",
            "type": "Ride",
            "name": "Synthetic Workout",
            "distance": 38000.0,
        },
    )
    focus = "how did my heart rate improve on hills in the last 3 months"
    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=detect_deep_query(focus), today=TODAY
    )

    assert len(context.activity_splits) == 38
    assert context.activity_splits[0].label == "km 1"
    assert context.activity_splits[-1].label == "km 38"


async def test_deep_extraction_paces_a_run_without_power_streams(settings: Settings) -> None:
    streams = {
        "time": list(range(601)),
        "distance": [index * 1000 / 300 for index in range(601)],
    }
    client = make_intervals_client(
        activities=[make_activity("fx-r", 5, activity_type="Run")],
        streams=streams,
        detail={
            "start_date_local": "2024-02-01T08:00:00",
            "type": "Run",
            "name": "Synthetic Run",
        },
    )
    focus = "how did my heart rate improve on hills in the last 3 months"
    extractor = DeepHistoricalExtractor(settings, client)
    context = await extractor.extract(focus, query=detect_deep_query(focus), today=TODAY)

    assert context.activity_splits[0].pace_s_per_km == 300
    assert context.activity_splits[0].average_speed_kmh is None
    assert not any(call[0] == "streams" and "watts" in call[2] for call in client.calls)


async def test_deep_extraction_prefers_the_referenced_activity(settings: Settings) -> None:
    activities = [make_activity("fx-a", 20), make_activity("fx-b", 5)]
    streams = {
        "time": list(range(301)),
        "distance": [index * 1000 / 300 for index in range(301)],
    }
    client = make_intervals_client(activities=activities, streams=streams)
    focus = "analyse my race on 2024-01-05 and how my heart rate held"
    query = detect_deep_query(focus, today=TODAY)
    assert query is not None and query.reference == date(2024, 1, 5)

    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=query, today=TODAY
    )

    assert ("detail", "fx-b") in client.calls
    assert any(call[0] == "streams" and call[1] == "fx-b" for call in client.calls)
    assert [split.label for split in context.activity_splits] == ["km 1"]


async def test_deep_extraction_prefers_the_cited_date_over_rank(settings: Settings) -> None:
    exact = make_activity("fx-exact", 5)
    exact["total_elevation_gain"] = 100.0
    neighbour = make_activity("fx-neighbour", 7)
    neighbour["total_elevation_gain"] = 2000.0
    client = make_intervals_client(activities=[neighbour, exact])
    focus = "analyse my run on 2024-01-05 and how my heart rate held on the hills"
    query = detect_deep_query(focus, today=TODAY)
    assert query is not None and query.reference == date(2024, 1, 5)

    await DeepHistoricalExtractor(settings, client).extract(focus, query=query, today=TODAY)

    assert ("detail", "fx-exact") in client.calls


async def test_deep_extraction_with_no_activity_near_the_cited_date(settings: Settings) -> None:
    client = make_intervals_client(activities=[make_activity("fx-far", 20)])
    focus = "analyse my run on 2024-01-05 and how my heart rate held"
    query = detect_deep_query(focus, today=TODAY)
    assert query is not None and query.reference == date(2024, 1, 5)

    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=query, today=TODAY
    )

    assert context.activity_detail is None
    assert context.activity_splits == []
    assert not any(call[0] == "detail" for call in client.calls)


async def test_deep_extraction_survives_missing_streams(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_intervals_client()

    async def boom(activity_id: str, types: object) -> dict[str, list]:
        raise IntervalsApiError(404, "no streams")

    monkeypatch.setattr(client, "get_activity_streams", boom)
    focus = "how did my heart rate improve on hills in the last 3 months"
    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=detect_deep_query(focus), today=TODAY
    )

    assert context.activity_detail is not None
    assert context.activity_splits == []


async def test_deep_extraction_survives_a_streams_value_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_intervals_client()

    async def boom(activity_id: str, types: object) -> dict[str, list]:
        raise ValueError("malformed streams")

    monkeypatch.setattr(client, "get_activity_streams", boom)
    focus = "how did my heart rate improve on hills in the last 3 months"
    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=detect_deep_query(focus), today=TODAY
    )

    assert context.activity_detail is not None
    assert context.activity_splits == []


async def test_deep_extraction_survives_an_unexpected_streams_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_intervals_client()

    async def boom(activity_id: str, types: object) -> dict[str, list]:
        raise RuntimeError("boom")

    monkeypatch.setattr(client, "get_activity_streams", boom)
    focus = "how did my heart rate improve on hills in the last 3 months"
    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=detect_deep_query(focus), today=TODAY
    )

    assert context.activity_detail is not None
    assert context.activity_splits == []


async def test_deep_extraction_survives_a_detail_failure(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_intervals_client()

    async def boom(activity_id: str, intervals: bool = True) -> dict[str, Any]:
        raise IntervalsApiError(500, "detail down")

    monkeypatch.setattr(client, "get_activity", boom)
    focus = "how did my heart rate improve on hills in the last 3 months"
    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=detect_deep_query(focus), today=TODAY
    )

    assert context.activity_detail is None
    assert context.activity_splits == []
    assert context.wellness != []


async def test_deep_extraction_keeps_the_detail_when_splits_fail(settings: Settings) -> None:
    streams: dict[str, list[Any]] = {
        "time": list(range(601)),
        "distance": [index * 1000 / 300 for index in range(601)],
        "watts": [1e308] * 601,
    }
    client = make_intervals_client(
        streams=streams,
        detail={
            "start_date_local": "2024-01-20T08:00:00",
            "type": "Ride",
            "name": "Huge power",
            "distance": 2000.0,
        },
    )
    focus = "how did my heart rate improve on hills in the last 3 months"
    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=detect_deep_query(focus), today=TODAY
    )

    assert context.activity_detail is not None
    assert context.activity_splits == []


async def test_standard_extraction_has_no_activity_splits(settings: Settings) -> None:
    client = make_intervals_client()
    extractor = StandardExtractor(settings, client)
    context = await extractor.extract("status check", today=TODAY)
    assert context.activity_splits == []
    assert not any(call[0] == "streams" for call in client.calls)


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
    assert len(context.wellness) == 2
    assert len(context.upcoming_events) == 2


async def test_budget_too_small_to_fit_focus_raises(settings: Settings) -> None:
    extractor = StandardExtractor(settings, make_intervals_client())
    with pytest.raises(RuntimeError, match="token budget"):
        await extractor.extract("status check", today=TODAY, max_tokens=1)


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


def test_hill_query_naming_a_ride_keeps_the_ride_family() -> None:
    query = detect_deep_query("heart rate improve on hilly bike sections")
    assert query is not None
    assert {"Ride", "VirtualRide", "GravelRide", "MountainBikeRide"} <= query.activity_types


def test_gravel_ride_named_in_a_hill_trend_keeps_the_ride_family() -> None:
    query = detect_deep_query("how did my heart rate improve on hilly gravel sections")
    assert query is not None
    assert "GravelRide" in query.activity_types


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


def test_recent_reference_within_the_standard_window_is_not_deep() -> None:
    assert detect_deep_query("what did I do on 2024-01-25?", today=date(2024, 2, 1)) is None


def test_reference_older_than_the_standard_window_is_deep() -> None:
    query = detect_deep_query("what did I do on 2024-01-18?", today=date(2024, 2, 1))
    assert query is not None
    assert query.lookback_days >= 21


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
    assert furthest.distance == 21097.5
    assert furthest.moving_time == 5400
    assert furthest.icu_training_load == 150


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
    extractor = StandardExtractor(settings, make_intervals_client())
    context = await extractor.extract("status check", today=TODAY)
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
    extractor = StandardExtractor(settings, client)
    context = await extractor.extract("status check", today=TODAY)
    assert [race.name for race in context.goal_races] == ["Coming Race"]


def test_budget_keeps_goal_races_while_other_sections_are_trimmed() -> None:
    race = GoalRace(
        event_id=1,
        name="Autumn Trail Race",
        date=date(2024, 2, 20),
        category="RACE_B",
        days_to_race=19,
        weeks_to_race=3,
        phase="Peak",
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


async def test_irregular_summary_spacing_degrades_to_no_rollup(settings: Settings) -> None:
    client = make_intervals_client(
        athlete_summary=[make_summary_week("2024-01-22"), make_summary_week("2024-01-24")]
    )
    extractor = StandardExtractor(settings, client)
    context = await extractor.extract("status check", today=TODAY)
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


async def test_deep_extraction_survives_a_non_finite_distance(settings: Settings) -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        streams: dict[str, list[Any]] = {
            "time": list(range(601)),
            "distance": [round(index * 1000 / 300, 4) for index in range(601)],
        }
        streams["distance"][300] = bad
        client = make_intervals_client(
            activities=[make_activity("fx-r", 5, activity_type="Run")],
            streams=streams,
            detail={
                "start_date_local": "2024-02-01T08:00:00",
                "type": "Run",
                "name": "Synthetic Run",
                "distance": 2000.0,
            },
        )
        focus = "how did my heart rate improve on hills in the last 3 months"
        context = await DeepHistoricalExtractor(settings, client).extract(
            focus, query=detect_deep_query(focus), today=TODAY
        )

        assert [split.label for split in context.activity_splits] == ["km 1", "km 2"]


async def test_deep_extraction_falls_back_to_the_stream_peak_without_a_distance(
    settings: Settings,
) -> None:
    streams: dict[str, list[Any]] = {
        "time": list(range(601)),
        "distance": [round(index * 1000 / 300, 4) for index in range(601)],
    }
    client = make_intervals_client(
        activities=[make_activity("fx-r", 5, activity_type="Run")],
        streams=streams,
        detail={
            "start_date_local": "2024-02-01T08:00:00",
            "type": "Run",
            "name": "Synthetic Run",
            "distance": 0.0,
        },
    )
    focus = "how did my heart rate improve on hills in the last 3 months"
    context = await DeepHistoricalExtractor(settings, client).extract(
        focus, query=detect_deep_query(focus), today=TODAY
    )

    assert [split.label for split in context.activity_splits] == ["km 1", "km 2"]


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


def test_day_month_with_a_relative_year_resolves_to_last_year() -> None:
    query = detect_deep_query(
        "Could you check the race of the 28th of september last year.",
        today=date(2026, 9, 16),
    )
    assert query is not None
    assert query.reference == date(2025, 9, 28)
    assert query.lookback_days >= 360


def test_bare_relative_year_keeps_the_generic_lookback() -> None:
    query = detect_deep_query("what did I do last year", today=date(2026, 9, 16))
    assert query is not None
    assert query.reference == date(2025, 9, 16)

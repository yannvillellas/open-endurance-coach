from datetime import date
from typing import Any

import pytest

from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.deep import DeepHistoricalExtractor, detect_deep_query
from open_endurance_coach.extractors.standard import StandardExtractor
from open_endurance_coach.schemas.context import CoachContext

TODAY = date(2024, 2, 1)


class FakeIntervalsClient:
    def __init__(
        self,
        activities: list[dict[str, Any]],
        wellness: list[dict[str, Any]],
        events: list[dict[str, Any]],
        sport_settings: list[dict[str, Any]],
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.activities = activities
        self.wellness = wellness
        self.events = events
        self.sport_settings = sport_settings
        self.detail = detail or {}
        self.calls: list[tuple[Any, ...]] = []

    async def list_activities(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        self.calls.append(("activities", oldest, newest))
        return list(self.activities)

    async def get_activity(self, activity_id: str, intervals: bool = True) -> dict[str, Any]:
        self.calls.append(("detail", activity_id))
        detail = dict(self.detail)
        detail["id"] = activity_id
        return detail

    async def list_wellness(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        self.calls.append(("wellness", oldest, newest))
        return list(self.wellness)

    async def list_events(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        self.calls.append(("events", oldest, newest))
        return list(self.events)

    async def get_sport_settings(self) -> list[dict[str, Any]]:
        self.calls.append(("sport_settings",))
        return list(self.sport_settings)


def make_activity(activity_id: str, day: int, activity_type: str = "Ride") -> dict[str, Any]:
    return {
        "id": activity_id,
        "start_date_local": f"2024-01-{day:02d}T08:00:00",
        "type": activity_type,
        "name": "Synthetic Workout",
        "icu_training_load": 80.0,
        "moving_time": 3600,
        "icu_average_watts": 240.0,
        "average_heartrate": 150.0,
        "total_elevation_gain": 800.0,
    }


def make_wellness(day: int) -> dict[str, Any]:
    return {"id": f"2024-01-{day:02d}", "ctl": 40.0, "hrv": 90.0, "sleepSecs": 28000}


def make_activity_list() -> list[dict[str, Any]]:
    return [
        make_activity("fx-a", 20),
        make_activity("fx-b", 18),
        make_activity("fx-c", 15, activity_type="Run"),
        make_activity("fx-d", 10),
        make_activity("fx-e", 5),
    ]


def make_client(**overrides: Any) -> FakeIntervalsClient:
    payloads: dict[str, Any] = {
        "activities": make_activity_list(),
        "wellness": [make_wellness(19), make_wellness(18), make_wellness(10)],
        "events": [
            {"name": "Tempo Session", "start_date_local": "2024-02-03T00:00:00"},
            {"name": "Long Ride", "start_date_local": "2024-02-10T00:00:00"},
        ],
        "sport_settings": [{"id": 1, "ftp": 250.0}],
        "detail": {
            "start_date_local": "2024-01-20T08:00:00",
            "type": "Ride",
            "name": "Synthetic Workout",
            "icu_intervals": [{"average_heartrate": 165}],
        },
    }
    payloads.update(overrides)
    return FakeIntervalsClient(**payloads)


async def test_standard_extraction_populates_all_sections(settings: Settings) -> None:
    extractor = StandardExtractor(settings, make_client())
    context = await extractor.extract("Analyze this week", today=TODAY)
    assert len(context.recent_activities) == 5
    assert len(context.wellness) == 3
    assert len(context.upcoming_events) == 2
    assert context.sport_settings[0].ftp == 250.0
    assert context.activity_detail is None
    assert context.user_feedback is None


async def test_standard_extraction_uses_expected_windows(settings: Settings) -> None:
    client = make_client()
    extractor = StandardExtractor(settings, client)
    await extractor.extract("status check", user_feedback="felt tired", today=TODAY)
    assert client.calls == [
        ("activities", "2024-01-18", "2024-02-02"),
        ("wellness", "2024-01-25", "2024-02-02"),
        ("events", "2024-02-01", "2024-02-15"),
        ("sport_settings",),
    ]


async def test_standard_extraction_keeps_newest_first(settings: Settings) -> None:
    extractor = StandardExtractor(settings, make_client())
    context = await extractor.extract("status check", today=TODAY)
    assert [item.id for item in context.recent_activities] == [
        "fx-a",
        "fx-b",
        "fx-c",
        "fx-d",
        "fx-e",
    ]


async def test_budget_overrun_trims_oldest_activities_first(settings: Settings) -> None:
    extractor = StandardExtractor(settings, make_client())
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
    extractor = StandardExtractor(settings, make_client())
    with pytest.raises(RuntimeError, match="token budget"):
        await extractor.extract("status check", today=TODAY, max_tokens=1)


@pytest.mark.parametrize(
    ("focus", "lookback", "metric"),
    [
        ("trend in heart rate over the last 6 weeks", 42, "heart_rate"),
        ("how did my power improve on hills in the last 3 months", 90, "elevation"),
        ("progress on climbs over the last 4 weeks", 28, "elevation"),
        ("heart rate evolution", 90, "heart_rate"),
    ],
)
def test_detect_deep_query_parses_focus(focus: str, lookback: int, metric: str) -> None:
    query = detect_deep_query(focus)
    assert query is not None
    assert query.lookback_days == lookback
    assert query.metric_focus == metric


def test_detect_deep_query_returns_none_for_plain_focus() -> None:
    assert detect_deep_query("status check") is None
    assert detect_deep_query("Analyze this week") is None


def test_climb_focus_restricts_to_rides() -> None:
    query = detect_deep_query("progress on climbs over the last 4 weeks")
    assert query is not None
    assert query.activity_types == frozenset({"Ride"})


async def test_deep_extraction_fetches_filtered_window_and_detail(settings: Settings) -> None:
    client = make_client(
        detail={
            "start_date_local": "2024-01-20T08:00:00",
            "type": "Ride",
            "name": "Synthetic Workout",
            "icu_intervals": [{"average_heartrate": 165}],
        }
    )
    extractor = DeepHistoricalExtractor(settings, client)
    context = await extractor.extract(
        "how did my heart rate improve on hills in the last 3 months", today=TODAY
    )
    activities_oldest = next(call for call in client.calls if call[0] == "activities")[1]
    assert activities_oldest == "2023-11-03"
    assert context.activity_detail is not None
    assert context.activity_detail.id == "fx-a"
    assert context.activity_detail.icu_intervals is not None
    assert len(context.recent_activities) == 4


async def test_deep_extraction_rejects_non_deep_focus(settings: Settings) -> None:
    extractor = DeepHistoricalExtractor(settings, make_client())
    with pytest.raises(ValueError, match="deep query"):
        await extractor.extract("status check", today=TODAY)


async def test_deep_extraction_respects_budget(settings: Settings) -> None:
    extractor = DeepHistoricalExtractor(settings, make_client())
    context = await extractor.extract("heart rate evolution", today=TODAY, max_tokens=150)
    assert isinstance(context, CoachContext)
    assert context.estimated_tokens() <= 150

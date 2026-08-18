import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from open_endurance_coach.schemas.intervals import Activity, Event, SportSettings, Wellness

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_activities_fixture_parses_into_activity_models() -> None:
    activities = [Activity.model_validate(item) for item in load_fixture("activities.json")]
    assert len(activities) == 19
    for activity in activities:
        assert activity.id.startswith("fx")
        assert isinstance(activity.start_date_local, datetime)
        assert activity.icu_intervals is None


def test_activity_detail_fixture_parses_with_intervals() -> None:
    activity = Activity.model_validate(load_fixture("activity_detail.json"))
    assert activity.type == "VirtualRide"
    assert activity.icu_training_load == 69
    assert activity.moving_time == 4552
    assert activity.icu_intervals is not None
    assert len(activity.icu_intervals) == 1
    interval = activity.icu_intervals[0]
    assert interval.average_heartrate == 153
    assert interval.max_heartrate == 183
    assert interval.zone == 2
    assert interval.end_time == 4553


def test_wellness_fixture_parses_into_wellness_models() -> None:
    rows = [Wellness.model_validate(item) for item in load_fixture("wellness.json")]
    assert len(rows) == 32
    first = rows[0]
    assert first.id == date(2024, 1, 1)
    assert first.ctl == 25.1609
    assert first.hrv == 119.1315
    assert first.restingHR == 55
    assert first.sleepSecs == 29550
    assert isinstance(first.updated, datetime)
    assert first.sportInfo is not None
    assert first.sportInfo[0].type == "Ride"


def test_sport_settings_fixture_parses_into_sport_settings_models() -> None:
    rows = [SportSettings.model_validate(item) for item in load_fixture("sport_settings.json")]
    assert len(rows) == 4
    first = rows[0]
    assert first.id == 10002
    assert first.athlete_id == "fx000003"
    assert first.ftp == 249.75
    assert first.types is not None and "Ride" in first.types
    assert first.power_zones is not None
    assert len(first.power_zones) == 7


def test_empty_events_fixture_parses() -> None:
    events = [Event.model_validate(item) for item in load_fixture("events.json")]
    assert events == []


def test_event_model_accepts_workout_payload() -> None:
    event = Event.model_validate(
        {
            "id": 10000,
            "name": "Tempo Session",
            "start_date_local": "2024-01-05T00:00:00",
            "category": "WORKOUT",
            "description": "3x10min sweet spot",
            "type": "Ride",
            "moving_time": 3600,
            "icu_training_load": 84,
            "workout_doc": {"steps": [{"duration": 600}]},
            "plan_folder_id": 42,
        }
    )
    assert event.category == "WORKOUT"
    assert event.workout_doc == {"steps": [{"duration": 600}]}


def test_activity_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        Activity.model_validate({"type": "Ride", "start_date_local": "2024-01-01T00:00:00"})


def test_wellness_rejects_missing_id() -> None:
    with pytest.raises(ValidationError):
        Wellness.model_validate({"ctl": 25.0})


def test_event_rejects_missing_name_or_date() -> None:
    with pytest.raises(ValidationError):
        Event.model_validate({"name": "Tempo Session"})
    with pytest.raises(ValidationError):
        Event.model_validate({"start_date_local": "2024-01-01T00:00:00"})


def test_sport_settings_rejects_missing_id() -> None:
    with pytest.raises(ValidationError):
        SportSettings.model_validate({"ftp": 290.0})


def test_wrong_field_types_are_rejected() -> None:
    activity = Activity.model_validate(
        {
            "id": "fx1",
            "start_date_local": "2024-01-01T00:00:00",
            "type": "Ride",
            "name": "Tempo Session",
        }
    )
    with pytest.raises(ValidationError):
        Activity.model_validate(
            {
                "id": "fx1",
                "start_date_local": "2024-01-01T00:00:00",
                "type": "Ride",
                "name": "Tempo Session",
                "icu_training_load": "high",
            }
        )
    with pytest.raises(ValidationError):
        Wellness.model_validate({"id": "2024-01-01", "ctl": "high"})
    with pytest.raises(ValidationError):
        Event.model_validate({"name": "x", "start_date_local": "not-a-date"})
    with pytest.raises(ValidationError):
        SportSettings.model_validate({"id": 1, "ftp": "fast"})
    assert activity.icu_training_load is None


def test_unknown_extra_fields_are_ignored() -> None:
    activity = Activity.model_validate(
        {
            "id": "fx1",
            "start_date_local": "2024-01-01T00:00:00",
            "type": "Ride",
            "name": "Tempo Session",
            "future_field": 123,
        }
    )
    assert not hasattr(activity, "future_field")

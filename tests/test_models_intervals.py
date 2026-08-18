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
    assert len(activities) > 0
    for activity in activities:
        assert activity.id.startswith("fx")
        assert isinstance(activity.start_date_local, datetime)
        assert activity.icu_intervals is None


def test_activity_detail_fixture_parses_with_intervals() -> None:
    activity = Activity.model_validate(load_fixture("activity_detail.json"))
    assert activity.type in {"Ride", "VirtualRide"}
    assert activity.icu_training_load is not None
    assert activity.moving_time is not None
    assert activity.icu_intervals is not None
    assert len(activity.icu_intervals) >= 1
    interval = activity.icu_intervals[0]
    assert isinstance(interval.average_heartrate, float)
    assert isinstance(interval.zone, int)
    assert interval.end_time is not None


def test_wellness_fixture_parses_into_wellness_models() -> None:
    rows = [Wellness.model_validate(item) for item in load_fixture("wellness.json")]
    assert len(rows) > 0
    first = rows[0]
    assert isinstance(first.id, date)
    assert isinstance(first.ctl, float)
    assert isinstance(first.hrv, float)
    assert isinstance(first.restingHR, int)
    assert isinstance(first.sleepSecs, int)
    assert isinstance(first.updated, datetime)
    assert first.sportInfo is not None
    assert first.sportInfo[0].type == "Ride"


def test_sport_settings_fixture_parses_into_sport_settings_models() -> None:
    rows = [SportSettings.model_validate(item) for item in load_fixture("sport_settings.json")]
    assert len(rows) > 0
    first = rows[0]
    assert isinstance(first.id, int)
    assert isinstance(first.athlete_id, str)
    assert isinstance(first.ftp, float)
    assert first.types is not None and "Ride" in first.types
    assert first.power_zones is not None
    assert len(first.power_zones) == 7


def test_events_fixture_parses_real_events() -> None:
    events = [Event.model_validate(item) for item in load_fixture("events.json")]
    assert len(events) == 2
    structured, minimal = events[0], events[1]
    for event in events:
        assert event.category == "WORKOUT"
        assert event.type == "Ride"
        assert event.start_date_local.time().isoformat() == "00:00:00"
    assert structured.description == "Recovery Spin"
    assert structured.moving_time == 2100
    assert structured.icu_training_load == 36
    assert structured.workout_doc is not None
    assert len(structured.workout_doc["steps"]) == 2
    repeats = structured.workout_doc["steps"][1]
    assert repeats["reps"] == 3
    assert repeats["steps"][0]["hr"] == {"units": "hr_zone", "value": 3}
    assert minimal.description is None
    assert minimal.moving_time is None
    assert minimal.icu_training_load is None
    assert minimal.workout_doc is not None
    assert minimal.workout_doc["steps"] == []


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

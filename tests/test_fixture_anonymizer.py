from typing import Any

from open_endurance_coach.fixtures.anonymize import (
    SYNTHETIC_VOCABULARY,
    anonymize_fixtures,
)


def anonymize(payload: Any) -> Any:
    return anonymize_fixtures({"payload.json": payload})["payload.json"]


def test_ids_are_remapped_and_consistent_across_payloads() -> None:
    fixtures = {
        "activities.json": [{"id": "i12345678", "athlete_id": 0, "event_id": "e123456"}],
        "detail.json": {"id": "i12345678", "parent_id": "e123456", "athlete_id": 12345},
    }
    result = anonymize_fixtures(fixtures)
    activity = result["activities.json"][0]
    detail = result["detail.json"]
    assert activity["id"] == detail["id"]
    assert activity["event_id"] == detail["parent_id"]
    assert activity["id"].startswith("fx")
    assert activity["athlete_id"] == 0
    assert detail["athlete_id"] == 10000


def test_dates_are_rewritten_to_synthetic_window_preserving_order() -> None:
    payload = {
        "dates": ["2026-08-01", "2026-08-10T08:30:00Z", "2026-08-10"],
        "events": [{"start_date_local": "2026-08-05"}, {"start_date_local": "2026-08-15"}],
    }
    result = anonymize(payload)
    first, second = result["dates"][0], result["dates"][1]
    assert first.startswith("2024")
    assert second.startswith("2024")
    assert result["dates"][1].endswith("Z")
    assert result["events"][0]["start_date_local"] < result["events"][1]["start_date_local"]


def test_epoch_timestamps_are_remapped_but_durations_are_not() -> None:
    payload = {
        "timestamps": [1754500000, 1754590000],
        "moving_time": 3600,
        "sleepSecs": 28000,
    }
    result = anonymize(payload)
    assert 1.7e9 < result["timestamps"][0] < 1.74e9
    assert result["timestamps"][0] < result["timestamps"][1]
    assert result["moving_time"] == 3600
    assert result["sleepSecs"] == 28000


def test_coordinates_are_offset_and_rounded() -> None:
    payload = {
        "start_latlng": [48.856614, 2.3522219],
        "lat": 48.123456,
        "lng": -2.987654,
    }
    result = anonymize(payload)
    for value in result["start_latlng"]:
        assert round(value, 2) == value
    assert result["start_latlng"][0] != 48.856614
    assert result["lat"] != 48.123456
    assert result["lng"] != -2.987654
    assert -90.0 <= result["lat"] <= 90.0
    assert -180.0 <= result["lng"] <= 180.0


def test_free_text_and_long_strings_are_replaced_from_vocabulary() -> None:
    payload = {
        "name": "Morning crit opener",
        "description": "A very personal and detailed description of the session",
        "notes": "felt tired after dinner",
        "long_field": "x" * 300,
    }
    result = anonymize(payload)
    for key in ("name", "description", "notes", "long_field"):
        assert result[key] in SYNTHETIC_VOCABULARY, key


def test_emails_and_urls_are_replaced() -> None:
    payload = {"email": "athlete@example.com", "url": "https://intervals.icu/activity/i12345678"}
    result = anonymize(payload)
    assert result["email"] == "fixture-email"
    assert result["url"] == "fixture-url"


def test_weight_and_ftp_are_jittered_within_bounds() -> None:
    payload = {"weight": 70.0, "ftp": 290.0}
    result = anonymize(payload)
    assert abs(result["weight"] - 70.0) / 70.0 <= 0.02
    assert result["weight"] != 70.0
    assert abs(result["ftp"] - 290.0) / 290.0 <= 0.02
    assert result["ftp"] != 290.0


def test_telemetry_values_and_structure_are_preserved() -> None:
    payload = {
        "avg_power": 250.0,
        "avg_hr": 152.0,
        "icu_training_load": 84.123456,
        "type": "Ride",
        "icu_intervals": [{"name": "Anaerobic Capacity", "duration": 30.0}],
    }
    result = anonymize(payload)
    assert result["avg_power"] == 250.0
    assert result["avg_hr"] == 152.0
    assert result["icu_training_load"] == 84.1235
    assert result["type"] == "Ride"
    assert len(result["icu_intervals"]) == 1
    assert result["icu_intervals"][0]["name"] == "Anaerobic Capacity"


def test_output_is_deterministic_for_same_input() -> None:
    payload = {
        "id": "i99999999",
        "start_date_local": "2026-08-17T10:00:00",
        "name": "Criterium race",
        "weight": 69.5,
    }
    assert anonymize(payload) == anonymize(payload)


def test_wellness_date_ids_are_treated_as_dates_not_ids() -> None:
    payload = {"id": "2026-08-16", "hrv": 78.0}
    result = anonymize(payload)
    assert result["id"].startswith("2024")
    assert result["hrv"] == 78.0


def test_group_tokens_are_scrambled_like_ids() -> None:
    fixtures = {
        "activities.json": [{"group": "1a74a9c1"}],
        "detail.json": {"group": "1a74a9c1"},
    }
    result = anonymize_fixtures(fixtures)
    assert result["activities.json"][0]["group"].startswith("fx")
    assert result["activities.json"][0]["group"] == result["detail.json"]["group"]

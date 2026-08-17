from typing import Any

import pytest

from open_endurance_coach.fixtures.record import record_fixtures


class FakeIntervalsClient:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self._payloads = payloads
        self.calls: list[str] = []
        self.activity_requests: list[str] = []

    async def list_activities(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        self.calls.append("list_activities")
        return self._payloads["activities"]

    async def get_activity(self, activity_id: str, intervals: bool = True) -> dict[str, Any]:
        self.calls.append("get_activity")
        self.activity_requests.append(activity_id)
        detail = dict(self._payloads["detail"])
        detail["id"] = activity_id
        return detail

    async def list_wellness(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        self.calls.append("list_wellness")
        return self._payloads["wellness"]

    async def list_events(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        self.calls.append("list_events")
        return self._payloads["events"]

    async def get_sport_settings(self) -> list[dict[str, Any]]:
        self.calls.append("get_sport_settings")
        return self._payloads["sport_settings"]

    async def get_athlete_summary(self) -> dict[str, Any]:
        self.calls.append("get_athlete_summary")
        return self._payloads["athlete_summary"]


def make_payloads() -> dict[str, Any]:
    return {
        "activities": [
            {
                "id": "i11111111",
                "start_date_local": "2026-08-17T08:00:00",
                "name": "real ride name",
            }
        ],
        "detail": {
            "id": "i11111111",
            "start_date_local": "2026-08-17T08:00:00",
            "icu_intervals": [{"name": "Anaerobic Capacity"}],
        },
        "wellness": [{"id": "2026-08-16", "hrv": 78.0}],
        "events": [{"id": 54321, "name": "real planned workout"}],
        "sport_settings": [{"sport": "Ride", "ftp": 290.0}],
        "athlete_summary": {"name": "Real Athlete Name"},
    }


async def test_record_fixtures_fetches_all_scopes_and_anonymizes(settings: Any) -> None:
    client = FakeIntervalsClient(make_payloads())
    fixtures = await record_fixtures(settings, client)
    assert set(fixtures) == {
        "activities.json",
        "activity_detail.json",
        "wellness.json",
        "events.json",
        "sport_settings.json",
        "athlete_summary.json",
    }
    activities = fixtures["activities.json"]
    assert activities[0]["id"].startswith("fx")
    assert activities[0]["name"] != "real ride name"
    assert fixtures["activity_detail.json"]["id"] == activities[0]["id"]
    assert fixtures["events.json"][0]["name"] != "real planned workout"
    assert fixtures["athlete_summary.json"]["name"] != "Real Athlete Name"
    assert fixtures["wellness.json"][0]["id"].startswith("2024")
    assert fixtures["activity_detail.json"]["icu_intervals"][0]["name"] == "Anaerobic Capacity"


async def test_record_fixtures_prefers_latest_ride_for_detail(settings: Any) -> None:
    payloads = make_payloads()
    payloads["activities"] = [
        {
            "id": "i00000002",
            "start_date_local": "2026-08-17T09:00:00",
            "name": "morning run",
            "type": "Run",
        },
        {
            "id": "i00000001",
            "start_date_local": "2026-08-16T18:00:00",
            "name": "evening ride",
            "type": "Ride",
        },
        {
            "id": "i11111111",
            "start_date_local": "2026-08-15T10:00:00",
            "name": "older ride",
            "type": "Ride",
        },
    ]
    client = FakeIntervalsClient(payloads)
    fixtures = await record_fixtures(settings, client)
    assert fixtures["activity_detail.json"]["id"] == fixtures["activities.json"][1]["id"]
    assert fixtures["activity_detail.json"]["id"] != fixtures["activities.json"][0]["id"]
    assert client.activity_requests == ["i00000001"]


async def test_record_fixtures_falls_back_to_latest_without_rides(settings: Any) -> None:
    payloads = make_payloads()
    payloads["activities"] = [
        {
            "id": "i00000002",
            "start_date_local": "2026-08-17T09:00:00",
            "name": "morning run",
            "type": "Run",
        },
        {
            "id": "i00000001",
            "start_date_local": "2026-08-16T18:00:00",
            "name": "evening run",
            "type": "Run",
        },
    ]
    client = FakeIntervalsClient(payloads)
    fixtures = await record_fixtures(settings, client)
    assert fixtures["activity_detail.json"]["id"] == fixtures["activities.json"][0]["id"]


async def test_record_fixtures_raises_when_no_activities(settings: Any) -> None:
    payloads = make_payloads()
    payloads["activities"] = []
    client = FakeIntervalsClient(payloads)
    with pytest.raises(RuntimeError, match="no activities"):
        await record_fixtures(settings, client)

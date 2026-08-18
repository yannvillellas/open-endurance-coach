import json
from datetime import datetime
from typing import Any

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.clients.llm import LlmCompletion


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


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


def make_activity(
    activity_id: str,
    day: int,
    activity_type: str = "Ride",
    name: str = "Synthetic Workout",
) -> dict[str, Any]:
    return {
        "id": activity_id,
        "start_date_local": f"2024-01-{day:02d}T08:00:00",
        "type": activity_type,
        "name": name,
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


def make_intervals_client(**overrides: Any) -> FakeIntervalsClient:
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


def make_event(
    event_id: int, day: str, name: str = "Tempo Session", category: str = "WORKOUT"
) -> dict[str, Any]:
    return {
        "id": event_id,
        "name": name,
        "start_date_local": f"{day}T00:00:00",
        "category": category,
        "type": "Ride",
    }


class FakeCalendarClient:
    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        self.events = [dict(item) for item in (events or [])]
        self.next_id = 20000
        self.created: list[dict[str, Any]] = []
        self.updated: list[tuple[str, dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.list_calls: list[tuple[str, str, str | None]] = []

    async def list_events(
        self, oldest: str, newest: str, category: str | None = None
    ) -> list[dict[str, Any]]:
        self.list_calls.append((oldest, newest, category))
        rows = []
        for event in self.events:
            if not (oldest <= event["start_date_local"][:10] < newest):
                continue
            if category and event.get("category") != category:
                continue
            rows.append(dict(event))
        return rows

    async def get_event(self, event_id: str) -> dict[str, Any]:
        for event in self.events:
            if str(event.get("id")) == str(event_id):
                return dict(event)
        raise IntervalsApiError(404, "not found")

    async def create_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        created = dict(payload)
        created["id"] = self.next_id
        self.next_id += 1
        self.events.append(created)
        self.created.append(created)
        return dict(created)

    async def update_event(self, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        for event in self.events:
            if str(event.get("id")) == str(event_id):
                event.update(payload)
                self.updated.append((event_id, dict(payload)))
                return dict(event)
        raise IntervalsApiError(404, "not found")

    async def delete_event(self, event_id: str) -> None:
        for index, event in enumerate(self.events):
            if str(event.get("id")) == str(event_id):
                self.deleted.append(event_id)
                del self.events[index]
                return
        raise IntervalsApiError(404, "not found")


class FakeLlmProvider:
    name = "fake"

    def __init__(self, responses: list[LlmCompletion] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        model: str,
        messages: list,
        thinking: bool,
        json_mode: bool,
        max_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None,
    ) -> LlmCompletion:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "thinking": thinking,
                "json_mode": json_mode,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            }
        )
        if not self.responses:
            raise AssertionError("FakeLlmProvider: no responses left")
        return self.responses.pop(0)

    async def aclose(self) -> None:
        pass


def completion(content: str, reasoning_content: str | None = None) -> LlmCompletion:
    return LlmCompletion(content=content, reasoning_content=reasoning_content, model="fake-model")


def report_json(summary: str = "Load stable.", **overrides: Any) -> str:
    payload: dict[str, Any] = {
        "summary": summary,
        "findings": ["Tempo block hit target."],
        "questions": ["RPE on Thursday?"],
        "mutations": [],
    }
    payload.update(overrides)
    return json.dumps(payload)

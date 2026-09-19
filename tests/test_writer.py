from datetime import date
from typing import Any, cast

import pytest

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.schemas.decisions import (
    CreateRace,
    CreateWorkout,
    DecisionReport,
    DeleteRace,
    DeleteWorkout,
    UpdateRace,
    UpdateWorkout,
)
from open_endurance_coach.store.records import Decision
from open_endurance_coach.writer.calendar import CalendarWriter, WriterError, _drift

from .fakes import FakeCalendarClient, make_event

DECIDED_AT = "2024-02-01T12:00:00+00:00"


def make_decision(*mutations: object) -> Decision:
    from datetime import datetime

    return Decision(
        id=1,
        draft_id=1,
        decided_at=datetime.fromisoformat(DECIDED_AT),
        applied_at=None,
        report=DecisionReport(summary="ok", mutations=list(mutations)),
    )


async def test_create_mutation_posts_minimal_payload() -> None:
    client = FakeCalendarClient()
    writer = CalendarWriter(client)
    mutation = CreateWorkout(
        action="create", name="Tempo Session", start_date_local=date(2024, 2, 5)
    )
    outcomes = await writer.apply_decision(make_decision(mutation))
    assert len(client.created) == 1
    assert client.created[0] == {
        "category": "WORKOUT",
        "name": "Tempo Session",
        "start_date_local": "2024-02-05T00:00:00",
        "id": 20000,
    }
    assert outcomes[0].target == "created"
    assert outcomes[0].event_id == 20000


async def test_create_mutation_passes_all_fields_through() -> None:
    client = FakeCalendarClient()
    writer = CalendarWriter(client)
    mutation = CreateWorkout(
        action="create",
        name="Sweet Spot",
        start_date_local=date(2024, 2, 6),
        description="3x10min sweet spot",
        type="Ride",
        moving_time=3600,
        distance=2000,
        icu_training_load=84.0,
    )
    await writer.apply_decision(make_decision(mutation))
    assert client.created[0] == {
        "category": "WORKOUT",
        "name": "Sweet Spot",
        "start_date_local": "2024-02-06T00:00:00",
        "description": "3x10min sweet spot",
        "type": "Ride",
        "moving_time": 3600,
        "time_target": 3600,
        "distance_target": 2000,
        "load_target": 84.0,
        "id": 20000,
    }


async def test_create_updates_existing_workout_with_same_name_and_date() -> None:
    client = FakeCalendarClient([make_event(10001, "2024-02-05", name="Tempo Session")])
    writer = CalendarWriter(client)
    mutation = CreateWorkout(
        action="create", name="Tempo Session", start_date_local=date(2024, 2, 5)
    )
    outcomes = await writer.apply_decision(make_decision(mutation))
    assert client.created == []
    assert client.updated == [
        (
            "10001",
            {
                "category": "WORKOUT",
                "name": "Tempo Session",
                "start_date_local": "2024-02-05T00:00:00",
            },
        )
    ]
    assert outcomes[0].target == "updated"
    assert outcomes[0].event_id == 10001


async def test_create_ignores_same_name_on_other_dates() -> None:
    client = FakeCalendarClient([make_event(10001, "2024-02-06", name="Tempo Session")])
    writer = CalendarWriter(client)
    mutation = CreateWorkout(
        action="create", name="Tempo Session", start_date_local=date(2024, 2, 5)
    )
    outcomes = await writer.apply_decision(make_decision(mutation))
    assert len(client.created) == 1
    assert outcomes[0].target == "created"


async def test_update_mutation_puts_only_changed_fields() -> None:
    client = FakeCalendarClient([make_event(10001, "2024-02-05")])
    writer = CalendarWriter(client)
    mutation = UpdateWorkout(action="update", event_id=10001, moving_time=4200)
    outcomes = await writer.apply_decision(make_decision(mutation))
    assert client.updated == [("10001", {"moving_time": 4200, "time_target": 4200})]
    assert outcomes[0].target == "updated"


async def test_update_mutation_formats_new_date() -> None:
    client = FakeCalendarClient([make_event(10001, "2024-02-05")])
    writer = CalendarWriter(client)
    mutation = UpdateWorkout(action="update", event_id=10001, start_date_local=date(2024, 2, 9))
    await writer.apply_decision(make_decision(mutation))
    assert client.updated == [("10001", {"start_date_local": "2024-02-09T00:00:00"})]


async def test_update_refuses_non_workout_event() -> None:
    client = FakeCalendarClient([make_event(10001, "2024-02-05", category="RACE_B")])
    writer = CalendarWriter(client)
    mutation = UpdateWorkout(action="update", event_id=10001, moving_time=4200)
    with pytest.raises(WriterError, match="non-WORKOUT"):
        await writer.apply_decision(make_decision(mutation))
    assert client.updated == []


async def test_update_missing_event_raises() -> None:
    client = FakeCalendarClient()
    writer = CalendarWriter(client)
    mutation = UpdateWorkout(action="update", event_id=10001, moving_time=4200)
    with pytest.raises(WriterError, match="not found"):
        await writer.apply_decision(make_decision(mutation))


async def test_delete_removes_workout_event() -> None:
    client = FakeCalendarClient([make_event(10001, "2024-02-05")])
    writer = CalendarWriter(client)
    mutation = DeleteWorkout(action="delete", event_id=10001)
    outcomes = await writer.apply_decision(make_decision(mutation))
    assert client.deleted == ["10001"]
    assert outcomes[0].target == "deleted"


async def test_delete_refuses_non_workout_event() -> None:
    client = FakeCalendarClient([make_event(10001, "2024-02-05", category="RACE_A")])
    writer = CalendarWriter(client)
    mutation = DeleteWorkout(action="delete", event_id=10001)
    with pytest.raises(WriterError, match="non-WORKOUT"):
        await writer.apply_decision(make_decision(mutation))
    assert client.deleted == []


async def test_delete_missing_event_is_skipped() -> None:
    client = FakeCalendarClient()
    writer = CalendarWriter(client)
    mutation = DeleteWorkout(action="delete", event_id=10001)
    outcomes = await writer.apply_decision(make_decision(mutation))
    assert client.deleted == []
    assert outcomes[0].target == "skipped"


async def test_mixed_decision_applies_in_order() -> None:
    client = FakeCalendarClient([make_event(10001, "2024-02-05")])
    writer = CalendarWriter(client)
    decision = make_decision(
        CreateWorkout(action="create", name="New Session", start_date_local=date(2024, 2, 7)),
        UpdateWorkout(action="update", event_id=10001, moving_time=4200),
        DeleteWorkout(action="delete", event_id=10001),
    )
    outcomes = await writer.apply_decision(decision)
    assert [outcome.action for outcome in outcomes] == ["create", "update", "delete"]
    assert len(client.created) == 1
    assert client.updated == [("10001", {"moving_time": 4200, "time_target": 4200})]
    assert client.deleted == ["10001"]


def make_race_create(**overrides: object) -> CreateRace:
    payload: dict[str, object] = {
        "action": "create_race",
        "name": "Autumn Trail Race",
        "start_date_local": date(2026, 9, 27),
        "category": "RACE_A",
    }
    payload.update(overrides)
    return CreateRace.model_validate(payload)


async def test_create_race_posts_race_category_payload() -> None:
    client = FakeCalendarClient()
    writer = CalendarWriter(client)
    mutation = make_race_create(type="Run")
    outcomes = await writer.apply_decision(make_decision(mutation))
    assert client.created[0] == {
        "category": "RACE_A",
        "name": "Autumn Trail Race",
        "start_date_local": "2026-09-27T00:00:00",
        "type": "Run",
        "id": 20000,
    }
    assert outcomes[0].action == "create_race"
    assert outcomes[0].target == "created"


async def test_create_race_updates_same_name_and_date_across_race_categories() -> None:
    client = FakeCalendarClient(
        [make_event(10001, "2026-09-27", name="Autumn Trail Race", category="RACE_B")]
    )
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(make_decision(make_race_create()))
    assert client.created == []
    assert client.updated == [
        (
            "10001",
            {
                "category": "RACE_A",
                "name": "Autumn Trail Race",
                "start_date_local": "2026-09-27T00:00:00",
            },
        )
    ]
    assert outcomes[0].target == "updated"
    assert outcomes[0].event_id == 10001


async def test_create_race_ignores_same_name_workout_event() -> None:
    client = FakeCalendarClient(
        [make_event(10001, "2026-09-27", name="Autumn Trail Race", category="WORKOUT")]
    )
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(make_decision(make_race_create()))
    assert len(client.created) == 1
    assert outcomes[0].target == "created"


async def test_update_race_puts_only_changed_fields_including_category() -> None:
    client = FakeCalendarClient([make_event(10001, "2026-09-27", category="RACE_B")])
    writer = CalendarWriter(client)
    mutation = UpdateRace(action="update_race", event_id=10001, category="RACE_A", moving_time=7200)
    outcomes = await writer.apply_decision(make_decision(mutation))
    assert client.updated == [("10001", {"category": "RACE_A", "moving_time": 7200})]
    assert outcomes[0].action == "update_race"
    assert outcomes[0].target == "updated"


async def test_update_race_refuses_non_race_event() -> None:
    client = FakeCalendarClient([make_event(10001, "2026-09-27", category="WORKOUT")])
    writer = CalendarWriter(client)
    mutation = UpdateRace(action="update_race", event_id=10001, category="RACE_A")
    with pytest.raises(WriterError, match="non-RACE"):
        await writer.apply_decision(make_decision(mutation))
    assert client.updated == []


async def test_update_race_missing_event_raises() -> None:
    client = FakeCalendarClient()
    writer = CalendarWriter(client)
    mutation = UpdateRace(action="update_race", event_id=10001, category="RACE_A")
    with pytest.raises(WriterError, match="not found"):
        await writer.apply_decision(make_decision(mutation))


async def test_delete_race_removes_race_event() -> None:
    client = FakeCalendarClient([make_event(10001, "2026-09-27", category="RACE_A")])
    writer = CalendarWriter(client)
    mutation = DeleteRace(action="delete_race", event_id=10001)
    outcomes = await writer.apply_decision(make_decision(mutation))
    assert client.deleted == ["10001"]
    assert outcomes[0].action == "delete_race"
    assert outcomes[0].target == "deleted"


async def test_delete_race_refuses_workout_event() -> None:
    client = FakeCalendarClient([make_event(10001, "2026-09-27", category="WORKOUT")])
    writer = CalendarWriter(client)
    mutation = DeleteRace(action="delete_race", event_id=10001)
    with pytest.raises(WriterError, match="non-RACE"):
        await writer.apply_decision(make_decision(mutation))
    assert client.deleted == []


async def test_delete_race_missing_event_is_skipped() -> None:
    client = FakeCalendarClient()
    writer = CalendarWriter(client)
    mutation = DeleteRace(action="delete_race", event_id=10001)
    outcomes = await writer.apply_decision(make_decision(mutation))
    assert client.deleted == []
    assert outcomes[0].target == "skipped"


async def test_mixed_workout_and_race_decision_applies_in_order() -> None:
    client = FakeCalendarClient(
        [make_event(10001, "2026-09-27", name="Autumn Trail Race", category="RACE_A")]
    )
    writer = CalendarWriter(client)
    decision = make_decision(
        CreateWorkout(action="create", name="Taper Opener", start_date_local=date(2026, 9, 22)),
        make_race_create(),
    )
    outcomes = await writer.apply_decision(decision)
    assert [outcome.action for outcome in outcomes] == ["create", "create_race"]
    assert len(client.created) == 1
    assert client.updated == [
        (
            "10001",
            {
                "category": "RACE_A",
                "name": "Autumn Trail Race",
                "start_date_local": "2026-09-27T00:00:00",
            },
        )
    ]


class _LeakyCategoryClient(FakeCalendarClient):
    """Returns the same event for any category filter, like a misbehaving server."""

    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__([event])
        self._event = event

    async def list_events(
        self, oldest: str, newest: str, category: str | None = None
    ) -> list[dict[str, Any]]:
        return [dict(self._event)]


async def test_create_workout_refuses_a_same_name_race_event() -> None:
    event = make_event(10001, "2024-02-05", name="Tempo Session", category="RACE_B")
    writer = CalendarWriter(_LeakyCategoryClient(event))
    mutation = CreateWorkout(
        action="create", name="Tempo Session", start_date_local=date(2024, 2, 5)
    )
    with pytest.raises(WriterError, match="non-WORKOUT"):
        await writer.apply_decision(make_decision(mutation))


async def test_create_race_refuses_a_same_name_workout_event() -> None:
    event = make_event(10001, "2024-02-05", name="Autumn Trail", category="WORKOUT")
    writer = CalendarWriter(_LeakyCategoryClient(event))
    mutation = CreateRace(
        action="create_race",
        name="Autumn Trail",
        start_date_local=date(2024, 2, 5),
        category="RACE_A",
    )
    with pytest.raises(WriterError, match="non-RACE"):
        await writer.apply_decision(make_decision(mutation))


async def test_create_workout_updates_a_leaked_category_less_event() -> None:
    client = _LeakyCategoryClient(
        {"id": 10001, "name": "Tempo Session", "start_date_local": "2026-09-22T00:00:00"}
    )
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(
        make_decision(
            CreateWorkout(action="create", name="Tempo Session", start_date_local=date(2026, 9, 22))
        )
    )
    assert client.created == []
    assert outcomes[0].target == "updated"


async def test_create_race_updates_a_leaked_category_less_event() -> None:
    client = _LeakyCategoryClient(
        {"id": 10001, "name": "Autumn Trail Race", "start_date_local": "2026-09-27T00:00:00"}
    )
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(make_decision(make_race_create()))
    assert client.created == []
    assert outcomes[0].target == "updated"


async def test_create_ignores_a_same_name_event_on_the_next_day() -> None:
    client = FakeCalendarClient(
        [make_event(10001, "2099-01-02", name="Tempo Session", category="WORKOUT")]
    )
    writer = CalendarWriter(client)
    outcome = await writer.apply_decision(
        make_decision(
            CreateWorkout(action="create", name="Tempo Session", start_date_local=date(2099, 1, 1))
        )
    )
    assert outcome[0].target == "created"
    assert [event["start_date_local"] for event in client.created] == ["2099-01-01T00:00:00"]


async def test_create_race_ignores_a_same_name_race_on_the_next_day() -> None:
    client = FakeCalendarClient(
        [make_event(10001, "2099-01-02", name="Autumn Trail Race", category="RACE_A")]
    )
    writer = CalendarWriter(client)
    outcome = await writer.apply_decision(
        make_decision(
            CreateRace(
                action="create_race",
                name="Autumn Trail Race",
                start_date_local=date(2099, 1, 1),
                category="RACE_B",
                moving_time=3600,
                icu_training_load=90,
            )
        )
    )
    assert outcome[0].target == "created"
    assert [event["start_date_local"] for event in client.created] == ["2099-01-01T00:00:00"]


async def test_create_race_payload_includes_distance() -> None:
    client = FakeCalendarClient()
    writer = CalendarWriter(client)
    await writer.apply_decision(
        make_decision(
            CreateRace(
                action="create_race",
                name="Autumn Trail Race",
                start_date_local=date(2099, 1, 1),
                category="RACE_B",
                type="TrailRun",
                moving_time=3728,
                distance=10900,
                icu_training_load=104,
            )
        )
    )
    assert client.created[0]["distance"] == 10900
    assert client.created[0]["type"] == "TrailRun"


async def test_update_race_payload_includes_distance() -> None:
    client = FakeCalendarClient(
        [make_event(10001, "2099-01-01", name="Autumn Trail Race", category="RACE_B")]
    )
    writer = CalendarWriter(client)
    await writer.apply_decision(
        make_decision(UpdateRace(action="update_race", event_id=10001, distance=10900))
    )
    assert client.updated[0][1]["distance"] == 10900


async def test_update_workout_payload_includes_targets() -> None:
    client = FakeCalendarClient([make_event(10001, "2024-02-05")])
    writer = CalendarWriter(client)
    await writer.apply_decision(
        make_decision(
            UpdateWorkout(
                action="update",
                event_id=10001,
                moving_time=6540,
                distance=5670,
                icu_training_load=42,
            )
        )
    )
    assert client.updated[0][1] == {
        "moving_time": 6540,
        "time_target": 6540,
        "distance_target": 5670,
        "load_target": 42,
    }


class _RecomputingCalendar(FakeCalendarClient):
    """Mimics Intervals recomputing duration and load from a parsed workout step."""

    async def create_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        created = await super().create_event(payload)
        self._recompute(created["id"])
        return created

    async def update_event(self, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        updated = await super().update_event(event_id, payload)
        self._recompute(event_id)
        return updated

    def _recompute(self, event_id: int | str) -> None:
        for event in self.events:
            if str(event.get("id")) == str(event_id):
                event["moving_time"] = 4464
                event["icu_training_load"] = 44
                event["load_target"] = 44


class _DistanceRewritingCalendar(FakeCalendarClient):
    async def update_event(self, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        updated = await super().update_event(event_id, payload)
        for event in self.events:
            if str(event.get("id")) == str(event_id):
                event["distance_target"] = 5670
        return updated


class _ReadBackFailingCalendar(FakeCalendarClient):
    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        super().__init__(events)
        self.fail_read_back = False

    async def create_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        created = await super().create_event(payload)
        self.fail_read_back = True
        return created

    async def update_event(self, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = await super().update_event(event_id, payload)
        self.fail_read_back = True
        return result

    async def get_event(self, event_id: str) -> dict[str, Any]:
        if self.fail_read_back:
            raise IntervalsApiError(500, "calendar unavailable")
        return await super().get_event(event_id)


class _IdlessCalendar(FakeCalendarClient):
    async def create_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {}


async def test_create_reports_duration_and_load_drift() -> None:
    client = _RecomputingCalendar()
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(
        make_decision(
            CreateWorkout(
                action="create",
                name="Hike Day 2",
                start_date_local=date(2026, 9, 22),
                type="Hike",
                moving_time=19680,
                icu_training_load=158,
            )
        )
    )
    assert outcomes[0].drift == [
        "moving_time stored 1h14m24s, requested 5h28m",
        "load stored 44, requested 158",
    ]


async def test_update_reports_distance_drift() -> None:
    client = _DistanceRewritingCalendar([make_event(10001, "2026-09-22")])
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(
        make_decision(UpdateWorkout(action="update", event_id=10001, distance=12400))
    )
    assert outcomes[0].drift == ["distance stored 5670m, requested 12400m"]


async def test_update_race_reports_drift() -> None:
    client = _RecomputingCalendar(
        [make_event(10001, "2099-01-01", name="Autumn Trail Race", category="RACE_B")]
    )
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(
        make_decision(
            UpdateRace(
                action="update_race",
                event_id=10001,
                moving_time=7200,
                icu_training_load=120,
            )
        )
    )
    assert outcomes[0].drift == [
        "moving_time stored 1h14m24s, requested 2h00m",
        "load stored 44, requested 120",
    ]


async def test_create_mutation_without_drift() -> None:
    client = FakeCalendarClient()
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(
        make_decision(
            CreateWorkout(
                action="create",
                name="Sweet Spot",
                start_date_local=date(2024, 2, 6),
                moving_time=3600,
                distance=2000,
                icu_training_load=84.0,
            )
        )
    )
    assert outcomes[0].drift == []


async def test_update_survives_a_failed_read_back() -> None:
    client = _ReadBackFailingCalendar([make_event(10001, "2024-02-05")])
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(
        make_decision(UpdateWorkout(action="update", event_id=10001, moving_time=4200))
    )
    assert outcomes[0].target == "updated"
    assert outcomes[0].drift == ["read-back failed; planned values were not verified"]


async def test_create_survives_a_failed_read_back() -> None:
    client = _ReadBackFailingCalendar()
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(
        make_decision(
            CreateWorkout(
                action="create",
                name="Session",
                start_date_local=date(2024, 2, 5),
                moving_time=3600,
            )
        )
    )
    assert outcomes[0].target == "created"
    assert outcomes[0].drift == ["read-back failed; planned values were not verified"]
    assert client.created


async def test_failed_read_back_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    client = _ReadBackFailingCalendar([make_event(10001, "2024-02-05")])
    writer = CalendarWriter(client)
    with caplog.at_level("WARNING"):
        await writer.apply_decision(
            make_decision(UpdateWorkout(action="update", event_id=10001, moving_time=4200))
        )
    assert "could not read event 10001 back" in caplog.text


async def test_create_without_an_id_reports_unverified() -> None:
    writer = CalendarWriter(_IdlessCalendar())
    outcomes = await writer.apply_decision(
        make_decision(
            CreateWorkout(
                action="create",
                name="Session",
                start_date_local=date(2024, 2, 5),
                moving_time=3600,
            )
        )
    )
    assert outcomes[0].event_id is None
    assert outcomes[0].drift == ["create returned no id; planned values were not verified"]


class _NonMappingReadBackCalendar(FakeCalendarClient):
    def __init__(self, events: list[dict[str, Any]] | None = None) -> None:
        super().__init__(events)
        self.after_update = False

    async def update_event(self, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = await super().update_event(event_id, payload)
        self.after_update = True
        return result

    async def get_event(self, event_id: str) -> dict[str, Any]:
        if self.after_update:
            return cast(dict[str, Any], [])
        return await super().get_event(event_id)


async def test_a_non_mapping_read_back_does_not_fail_the_write() -> None:
    writer = CalendarWriter(_NonMappingReadBackCalendar([make_event(10001, "2024-02-05")]))
    outcomes = await writer.apply_decision(
        make_decision(UpdateWorkout(action="update", event_id=10001, moving_time=4200))
    )
    assert outcomes[0].target == "updated"
    assert outcomes[0].drift == ["read-back failed; planned values were not verified"]


class _ComputedDistanceCalendar(FakeCalendarClient):
    async def update_event(self, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        updated = await super().update_event(event_id, payload)
        for event in self.events:
            if str(event.get("id")) == str(event_id):
                event["distance"] = 5670
        return updated


async def test_update_reports_a_parsed_distance_step() -> None:
    client = _ComputedDistanceCalendar([make_event(10001, "2026-09-22")])
    writer = CalendarWriter(client)
    outcomes = await writer.apply_decision(
        make_decision(UpdateWorkout(action="update", event_id=10001, distance=8690))
    )
    assert outcomes[0].drift == ["computed distance stored 5670m, target 8690m"]


def test_drift_reads_the_target_fields_per_event_kind() -> None:
    workout = UpdateWorkout(
        action="update",
        event_id=10001,
        moving_time=3600,
        distance=12400,
        icu_training_load=158,
    )
    stored = {"moving_time": "3600.0", "distance_target": "12400.0", "load_target": "158"}
    assert _drift(stored, workout) == []
    race = UpdateRace(action="update_race", event_id=10001, distance=10900, moving_time=7200)
    race_stored = {"moving_time": 7200, "distance": 10900}
    assert _drift(race_stored, race) == []
    short = UpdateWorkout(action="update", event_id=10001, moving_time=60)
    assert _drift({"moving_time": 44}, short) == ["moving_time stored 44s, requested 1m"]
    parsed_wrong = UpdateWorkout(action="update", event_id=10001, moving_time=6540)
    assert _drift({"moving_time": 3649}, parsed_wrong) == [
        "moving_time stored 1h00m49s, requested 1h49m"
    ]


class _ServerErrorCalendar(FakeCalendarClient):
    async def get_event(self, event_id: str) -> dict[str, Any]:
        raise IntervalsApiError(500, "calendar unavailable")


async def test_update_does_not_swallow_a_server_error_on_lookup() -> None:
    client = _ServerErrorCalendar()
    writer = CalendarWriter(client)
    with pytest.raises(IntervalsApiError):
        await writer.apply_decision(
            make_decision(UpdateWorkout(action="update", event_id=10001, name="Renamed"))
        )
    assert client.updated == []

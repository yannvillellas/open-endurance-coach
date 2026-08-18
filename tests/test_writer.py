from datetime import date

import pytest

from open_endurance_coach.schemas.decisions import (
    CreateWorkout,
    DecisionReport,
    DeleteWorkout,
    UpdateWorkout,
)
from open_endurance_coach.store.records import Decision
from open_endurance_coach.writer.calendar import CalendarWriter, WriterError

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
        "icu_training_load": 84.0,
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
    assert client.updated == [("10001", {"moving_time": 4200})]
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
    assert client.updated == [("10001", {"moving_time": 4200})]
    assert client.deleted == ["10001"]


async def test_dry_run_makes_no_writes() -> None:
    client = FakeCalendarClient([make_event(10001, "2024-02-05")])
    writer = CalendarWriter(client)
    decision = make_decision(
        CreateWorkout(action="create", name="New Session", start_date_local=date(2024, 2, 7)),
        UpdateWorkout(action="update", event_id=10001, moving_time=4200),
        DeleteWorkout(action="delete", event_id=10001),
    )
    outcomes = await writer.apply_decision(decision, dry_run=True)
    assert [outcome.target for outcome in outcomes] == ["created", "updated", "deleted"]
    assert client.created == []
    assert client.updated == []
    assert client.deleted == []

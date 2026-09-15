from datetime import date

import pytest
from pydantic import ValidationError

from open_endurance_coach.schemas.decisions import (
    CreateRace,
    CreateWorkout,
    DecisionReport,
    DeleteRace,
    DeleteWorkout,
    UpdateRace,
    UpdateWorkout,
)

CREATE_PAYLOAD = {
    "action": "create",
    "name": "Tempo Session",
    "start_date_local": "2024-01-05",
    "description": "3x10min sweet spot",
    "type": "Ride",
    "moving_time": 3600,
    "icu_training_load": 84,
}

UPDATE_PAYLOAD = {
    "action": "update",
    "event_id": 10001,
    "moving_time": 4200,
}

DELETE_PAYLOAD = {
    "action": "delete",
    "event_id": "e10001",
}

CREATE_RACE_PAYLOAD = {
    "action": "create_race",
    "name": "Autumn Trail Race",
    "start_date_local": "2026-09-27",
    "category": "RACE_A",
    "type": "Run",
}

UPDATE_RACE_PAYLOAD = {
    "action": "update_race",
    "event_id": 20001,
    "category": "RACE_B",
}

DELETE_RACE_PAYLOAD = {
    "action": "delete_race",
    "event_id": "e20001",
}


def test_valid_report_with_all_mutation_kinds_parses() -> None:
    report = DecisionReport.model_validate(
        {
            "summary": "Execution matched targets; keep load stable.",
            "findings": ["FTP sessions executed above target power"],
            "questions": ["What was your RPE on Tuesday's ride?"],
            "mutations": [CREATE_PAYLOAD, UPDATE_PAYLOAD, DELETE_PAYLOAD],
        }
    )
    assert report.summary == "Execution matched targets; keep load stable."
    assert len(report.mutations) == 3
    assert isinstance(report.mutations[0], CreateWorkout)
    assert isinstance(report.mutations[1], UpdateWorkout)
    assert isinstance(report.mutations[2], DeleteWorkout)


def test_valid_report_with_no_mutations_parses() -> None:
    report = DecisionReport.model_validate({"summary": "No calendar changes needed."})
    assert report.findings == []
    assert report.questions == []
    assert report.mutations == []


def test_create_mutation_parses_full_payload() -> None:
    mutation = CreateWorkout.model_validate(CREATE_PAYLOAD)
    assert mutation.name == "Tempo Session"
    assert mutation.start_date_local == date(2024, 1, 5)
    assert mutation.moving_time == 3600


def test_update_accepts_int_or_str_event_id() -> None:
    assert UpdateWorkout.model_validate(UPDATE_PAYLOAD).event_id == 10001
    assert (
        UpdateWorkout.model_validate(
            {"action": "update", "event_id": "e10001", "name": "New Name"}
        ).event_id
        == "e10001"
    )


def test_report_rejects_missing_or_empty_summary() -> None:
    with pytest.raises(ValidationError):
        DecisionReport.model_validate({})
    with pytest.raises(ValidationError):
        DecisionReport.model_validate({"summary": ""})


def test_unknown_mutation_action_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DecisionReport.model_validate(
            {"summary": "x", "mutations": [{"action": "replace", "event_id": 1}]}
        )


def test_create_requires_name_and_date() -> None:
    with pytest.raises(ValidationError):
        CreateWorkout.model_validate({"action": "create", "start_date_local": "2024-01-05"})
    with pytest.raises(ValidationError):
        CreateWorkout.model_validate({"action": "create", "name": "Tempo Session"})
    with pytest.raises(ValidationError):
        CreateWorkout.model_validate(
            {"action": "create", "name": "Tempo Session", "start_date_local": "not-a-date"}
        )


def test_update_requires_event_id_and_at_least_one_change() -> None:
    with pytest.raises(ValidationError):
        UpdateWorkout.model_validate({"action": "update", "moving_time": 3600})
    with pytest.raises(ValidationError):
        UpdateWorkout.model_validate({"action": "update", "event_id": 1})


def test_delete_requires_event_id() -> None:
    with pytest.raises(ValidationError):
        DeleteWorkout.model_validate({"action": "delete"})


def test_unknown_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        DecisionReport.model_validate({"summary": "x", "hallucinated_field": 1})
    with pytest.raises(ValidationError):
        CreateWorkout.model_validate({**CREATE_PAYLOAD, "hallucinated_field": 1})
    with pytest.raises(ValidationError):
        UpdateWorkout.model_validate({**UPDATE_PAYLOAD, "hallucinated_field": 1})
    with pytest.raises(ValidationError):
        DeleteWorkout.model_validate({**DELETE_PAYLOAD, "hallucinated_field": 1})


def test_wrong_field_types_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CreateWorkout.model_validate({**CREATE_PAYLOAD, "moving_time": "1h"})
    with pytest.raises(ValidationError):
        CreateWorkout.model_validate({**CREATE_PAYLOAD, "icu_training_load": "high"})
    with pytest.raises(ValidationError):
        DecisionReport.model_validate({"summary": "x", "findings": "not a list"})


def test_report_parses_workout_and_race_mutations_together() -> None:
    report = DecisionReport.model_validate(
        {
            "summary": "Taper week into an A race.",
            "mutations": [
                CREATE_PAYLOAD,
                CREATE_RACE_PAYLOAD,
                UPDATE_RACE_PAYLOAD,
                DELETE_PAYLOAD,
            ],
        }
    )
    assert len(report.mutations) == 4
    assert isinstance(report.mutations[0], CreateWorkout)
    assert isinstance(report.mutations[1], CreateRace)
    assert isinstance(report.mutations[2], UpdateRace)
    assert isinstance(report.mutations[3], DeleteWorkout)


def test_race_mutations_parse_priority_categories() -> None:
    for category in ("RACE_A", "RACE_B", "RACE_C"):
        mutation = CreateRace.model_validate({**CREATE_RACE_PAYLOAD, "category": category})
        assert mutation.category == category
        assert mutation.start_date_local == date(2026, 9, 27)


def test_create_race_requires_name_date_and_category() -> None:
    with pytest.raises(ValidationError):
        CreateRace.model_validate({"action": "create_race", "start_date_local": "2026-09-27"})
    with pytest.raises(ValidationError):
        CreateRace.model_validate({"action": "create_race", "name": "R", "category": "RACE_A"})
    with pytest.raises(ValidationError):
        CreateRace.model_validate({**CREATE_RACE_PAYLOAD, "category": "RACE_D"})


def test_update_race_requires_event_id_and_at_least_one_change() -> None:
    with pytest.raises(ValidationError):
        UpdateRace.model_validate({"action": "update_race", "category": "RACE_A"})
    with pytest.raises(ValidationError):
        UpdateRace.model_validate({"action": "update_race", "event_id": 1})


def test_delete_race_requires_event_id() -> None:
    with pytest.raises(ValidationError):
        DeleteRace.model_validate({"action": "delete_race"})


def test_race_mutation_rejects_non_race_category() -> None:
    with pytest.raises(ValidationError):
        CreateRace.model_validate({**CREATE_RACE_PAYLOAD, "category": "WORKOUT"})
    with pytest.raises(ValidationError):
        UpdateRace.model_validate({**UPDATE_RACE_PAYLOAD, "category": "WORKOUT"})
    with pytest.raises(ValidationError):
        DecisionReport.model_validate(
            {
                "summary": "x",
                "mutations": [{**CREATE_RACE_PAYLOAD, "category": "WORKOUT"}],
            }
        )


def test_workout_mutation_rejects_race_category() -> None:
    with pytest.raises(ValidationError):
        CreateWorkout.model_validate({**CREATE_PAYLOAD, "category": "RACE_A"})
    with pytest.raises(ValidationError):
        UpdateWorkout.model_validate({**UPDATE_PAYLOAD, "category": "RACE_B"})
    with pytest.raises(ValidationError):
        DeleteWorkout.model_validate({**DELETE_PAYLOAD, "category": "RACE_C"})


def test_race_mutations_reject_extras_and_wrong_action() -> None:
    with pytest.raises(ValidationError):
        CreateRace.model_validate({**CREATE_RACE_PAYLOAD, "hallucinated_field": 1})
    with pytest.raises(ValidationError):
        UpdateRace.model_validate({**UPDATE_RACE_PAYLOAD, "hallucinated_field": 1})
    with pytest.raises(ValidationError):
        DeleteRace.model_validate({**DELETE_RACE_PAYLOAD, "hallucinated_field": 1})
    with pytest.raises(ValidationError):
        CreateRace.model_validate({**CREATE_RACE_PAYLOAD, "action": "create"})
    with pytest.raises(ValidationError):
        DecisionReport.model_validate(
            {
                "summary": "x",
                "mutations": [{"action": "replace_race", "event_id": 1}],
            }
        )


def test_report_round_trips_through_json_with_race_mutations() -> None:
    report = DecisionReport.model_validate(
        {
            "summary": "Build week into an A race.",
            "mutations": [
                CREATE_PAYLOAD,
                CREATE_RACE_PAYLOAD,
                UPDATE_RACE_PAYLOAD,
                DELETE_RACE_PAYLOAD,
            ],
        }
    )
    reparsed = DecisionReport.model_validate_json(report.model_dump_json())
    assert reparsed == report
    assert [mutation.action for mutation in reparsed.mutations] == [
        "create",
        "create_race",
        "update_race",
        "delete_race",
    ]


def test_update_race_accepts_each_field() -> None:
    per_field: list[dict[str, str | int | float]] = [
        {"name": "Renamed Race"},
        {"start_date_local": "2026-10-04"},
        {"category": "RACE_C"},
        {"description": "hilly loop"},
        {"type": "Ride"},
        {"moving_time": 7200},
        {"icu_training_load": 150},
    ]
    for fields in per_field:
        mutation = UpdateRace.model_validate({"action": "update_race", "event_id": 1, **fields})
        assert mutation.event_id == 1

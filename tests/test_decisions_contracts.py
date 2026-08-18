from datetime import date

import pytest
from pydantic import ValidationError

from open_endurance_coach.schemas.decisions import (
    CreateWorkout,
    DecisionReport,
    DeleteWorkout,
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

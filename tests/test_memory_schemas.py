import json
from datetime import date

import pytest
from pydantic import ValidationError

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport, MemoryOp
from open_endurance_coach.schemas.memory import MemoryAction, MemoryCategory, MemoryOrigin


def test_remember_op_round_trips() -> None:
    op = MemoryOp.model_validate(
        {
            "action": "remember",
            "category": "training_constraints",
            "origin": "declared",
            "text": "Pas de vélo jusqu'au 20/10",
            "expires_on": "2026-10-20",
            "trigger_keywords": ["côtes"],
            "trigger_on": "2026-10-01",
        }
    )

    assert op.action is MemoryAction.REMEMBER
    assert op.category is MemoryCategory.TRAINING_CONSTRAINTS
    assert op.origin is MemoryOrigin.DECLARED
    assert op.id is None
    assert op.expires_on == date(2026, 10, 20)
    assert op.trigger_keywords == ["côtes"]
    assert op.trigger_on == date(2026, 10, 1)


def test_update_op_requires_id_and_text() -> None:
    op = MemoryOp.model_validate(
        {
            "action": "update",
            "id": 12,
            "text": "Pas de vélo jusqu'au 20/10",
            "expires_on": "2026-10-20",
        }
    )

    assert op.action is MemoryAction.UPDATE
    assert op.id == 12
    assert op.text == "Pas de vélo jusqu'au 20/10"
    assert op.expires_on == date(2026, 10, 20)


def test_close_and_forget_require_only_an_id() -> None:
    close = MemoryOp.model_validate({"action": "close", "id": 7})
    forget = MemoryOp.model_validate({"action": "forget", "id": 8})

    assert close.action is MemoryAction.CLOSE
    assert forget.action is MemoryAction.FORGET


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"action": "remember", "origin": "declared", "text": "x"}, "category"),
        ({"action": "remember", "category": "injury_niggle", "text": "x"}, "origin"),
        ({"action": "remember", "category": "injury_niggle", "origin": "declared"}, "text"),
        (
            {
                "action": "remember",
                "id": 3,
                "category": "injury_niggle",
                "origin": "declared",
                "text": "x",
            },
            "id",
        ),
        ({"action": "update", "text": "x"}, "id"),
        ({"action": "update", "id": 3}, "text"),
        ({"action": "close"}, "id"),
        ({"action": "forget", "id": 3, "text": "x"}, "content"),
        ({"action": "close", "id": 3, "category": "injury_niggle"}, "category"),
        ({"action": "remember", "category": "mood", "origin": "declared", "text": "x"}, "category"),
        (
            {"action": "remember", "category": "injury_niggle", "origin": "guessed", "text": "x"},
            "origin",
        ),
        ({"action": "forget", "id": 3, "extra": True}, "extra"),
        (
            {
                "action": "remember",
                "category": "injury_niggle",
                "origin": "declared",
                "text": "x\n y",
            },
            "line",
        ),
        (
            {"action": "remember", "category": "injury_niggle", "origin": "declared", "text": "  "},
            "text",
        ),
    ],
)
def test_memory_op_rejects_malformed_shapes(payload: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        MemoryOp.model_validate(payload)


def test_decision_report_parses_memory_ops() -> None:
    report = DecisionReport.model_validate(
        {
            "summary": "Noted",
            "memory": [
                {
                    "action": "remember",
                    "category": "injury_niggle",
                    "origin": "declared",
                    "text": "Tendinite genou gauche",
                }
            ],
        }
    )

    assert [op.action for op in report.memory] == [MemoryAction.REMEMBER]


def test_decision_report_without_memory_stays_valid() -> None:
    report = DecisionReport.model_validate({"summary": "No facts"})
    assert report.memory == []


def test_memory_ops_survive_a_json_round_trip() -> None:
    report = DecisionReport.model_validate(
        {
            "summary": "Noted",
            "memory": [
                {
                    "action": "remember",
                    "category": "injury_niggle",
                    "origin": "declared",
                    "text": "Genou gauche",
                },
                {
                    "action": "update",
                    "id": 1,
                    "text": "Genou gauche, revu",
                    "expires_on": "2026-10-20",
                },
                {"action": "close", "id": 2},
                {"action": "forget", "id": 3},
            ],
        }
    )

    reloaded = DecisionReport.model_validate(json.loads(report.model_dump_json()))

    assert reloaded.memory == report.memory


def test_context_renders_the_memory_sections() -> None:
    context = CoachContext.model_validate(
        {
            "focus": "plan the week",
            "coach_memory": [
                {
                    "id": 1,
                    "category": "injury_niggle",
                    "origin": "declared",
                    "text": "Tendinite genou gauche",
                    "valid_from": "2026-09-01",
                }
            ],
            "coach_memory_index": [
                {"id": 2, "category": "athlete_profile", "label": "42 ans, 10 ans de vélo"}
            ],
        }
    )

    sections = context.sections()
    assert sections["coach_memory"] == [
        {
            "id": 1,
            "category": "injury_niggle",
            "origin": "declared",
            "text": "Tendinite genou gauche",
            "valid_from": "2026-09-01",
        }
    ]
    assert sections["coach_memory_index"] == [
        {"id": 2, "category": "athlete_profile", "label": "42 ans, 10 ans de vélo"}
    ]
    assert context.data_tokens() > 0


def test_memory_sections_are_optional() -> None:
    context = CoachContext.model_validate({"focus": "status check"})
    assert context.coach_memory == []
    assert context.coach_memory_index == []
    assert "coach_memory" not in context.sections()

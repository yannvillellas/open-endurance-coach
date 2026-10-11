import logging
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from open_endurance_coach.config import Settings
from open_endurance_coach.schemas.memory import MemoryStatus

from .fakes import FakeLlmProvider, completion, make_engine, report_json


def memory_op(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": "remember",
        "category": "training_constraints",
        "origin": "declared",
        "text": "Pas de vélo jusqu'au 20/10",
    }
    payload.update(overrides)
    return payload


async def test_analyze_remembers_a_fact_and_reports_a_notice(
    settings: Settings, tmp_path: Path
) -> None:
    provider = FakeLlmProvider([completion(report_json(memory=[memory_op()]))])
    engine, store = make_engine(settings, tmp_path, provider)
    notices: list[str] = []

    await engine.analyze("no bike for now", memory_notices=notices)

    facts = store.list_active_facts()
    assert len(facts) == 1
    assert facts[0].text == "Pas de vélo jusqu'au 20/10"
    assert facts[0].source_message_id is not None
    assert notices == ["Remembered: [training_constraints] Pas de vélo jusqu'au 20/10"]


async def test_memory_is_applied_even_when_the_plan_is_blocked(
    settings: Settings, tmp_path: Path
) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(needs_input=["Until when?"], memory=[memory_op()]))]
    )
    engine, store = make_engine(settings, tmp_path, provider)
    notices: list[str] = []

    await engine.analyze("no bike for now", memory_notices=notices)

    assert len(store.list_active_facts()) == 1
    assert notices == ["Remembered: [training_constraints] Pas de vélo jusqu'au 20/10"]


async def test_update_replaces_the_payload_in_place(settings: Settings, tmp_path: Path) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(memory=[memory_op(expires_on="2099-01-01")])),
            completion(
                report_json(
                    memory=[{"action": "update", "id": 1, "text": "Pas de vélo jusqu'au 21/10"}]
                )
            ),
            completion(
                report_json(
                    memory=[
                        {
                            "action": "update",
                            "id": 1,
                            "text": "Pas de vélo jusqu'au 22/10",
                            "expires_on": "2099-02-01",
                        }
                    ]
                )
            ),
        ]
    )
    engine, store = make_engine(settings, tmp_path, provider)

    await engine.analyze("first", memory_notices=[])
    await engine.analyze("second", memory_notices=[])

    fact = store.list_active_facts()[0]
    assert fact.id == 1
    assert fact.text == "Pas de vélo jusqu'au 21/10"
    assert fact.expires_on is None

    await engine.analyze("third", memory_notices=[])

    fact = store.list_active_facts()[0]
    assert fact.text == "Pas de vélo jusqu'au 22/10"
    assert fact.expires_on == date(2099, 2, 1)


async def test_close_keeps_a_history_fact_and_deletes_a_constraint(
    settings: Settings, tmp_path: Path
) -> None:
    provider = FakeLlmProvider(
        [
            completion(
                report_json(
                    memory=[
                        memory_op(category="injury_niggle", text="Tendinite genou gauche"),
                        memory_op(text="Pas de vélo"),
                    ]
                )
            ),
            completion(
                report_json(
                    memory=[
                        {"action": "close", "id": 1},
                        {"action": "close", "id": 2},
                    ]
                )
            ),
        ]
    )
    engine, store = make_engine(settings, tmp_path, provider)
    notices: list[str] = []

    await engine.analyze("first", memory_notices=[])
    await engine.analyze("second", memory_notices=notices)

    injury = store.get_fact(1)
    assert injury is not None
    assert injury.status is MemoryStatus.RESOLVED
    assert store.get_fact(2) is None
    assert notices == [
        "Closed: [injury_niggle] Tendinite genou gauche (kept as history)",
        "Closed: [training_constraints] Pas de vélo",
    ]


async def test_forget_deletes_a_history_fact(settings: Settings, tmp_path: Path) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(memory=[memory_op(category="health_medical", text="Asthme")])),
            completion(report_json(memory=[{"action": "forget", "id": 1}])),
        ]
    )
    engine, store = make_engine(settings, tmp_path, provider)
    notices: list[str] = []

    await engine.analyze("first", memory_notices=[])
    await engine.analyze("second", memory_notices=notices)

    assert store.get_fact(1) is None
    assert notices == ["Forgotten: [health_medical] Asthme"]


async def test_an_unknown_id_is_skipped_with_a_warning(
    settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeLlmProvider(
        [
            completion(
                report_json(
                    memory=[
                        {"action": "forget", "id": 999},
                        memory_op(),
                    ]
                )
            )
        ]
    )
    engine, store = make_engine(settings, tmp_path, provider)
    notices: list[str] = []

    with caplog.at_level(logging.WARNING):
        await engine.analyze("first", memory_notices=notices)

    assert "memory operation skipped" in caplog.text
    assert len(store.list_active_facts()) == 1
    assert notices == ["Remembered: [training_constraints] Pas de vélo jusqu'au 20/10"]


async def test_feedback_applies_memory_and_returns_notices(
    settings: Settings, tmp_path: Path
) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json()),
            completion(report_json(intent="chat", memory=[memory_op()])),
        ]
    )
    engine, store = make_engine(settings, tmp_path, provider)

    proposal = await engine.analyze("hello", memory_notices=[])
    outcome = await engine.submit_feedback(proposal.id, "no bike until the 20th")

    assert outcome.memory_notices == (
        "Remembered: [training_constraints] Pas de vélo jusqu'au 20/10",
    )
    assert len(store.list_active_facts()) == 1


async def test_expire_memory_applies_the_category_policy(
    settings: Settings, tmp_path: Path
) -> None:
    provider = FakeLlmProvider(
        [
            completion(
                report_json(
                    memory=[
                        memory_op(category="injury_niggle", text="Genou", expires_on="2000-01-01"),
                        memory_op(expires_on="2000-01-01"),
                        memory_op(category="injury_niggle", text="Épaule"),
                    ]
                )
            )
        ]
    )
    engine, store = make_engine(settings, tmp_path, provider)
    await engine.analyze("first", memory_notices=[])

    outcomes = engine.expire_memory()

    assert outcomes == [(1, "resolved"), (2, "deleted")]
    kept = store.get_fact(1)
    assert kept is not None
    assert kept.status is MemoryStatus.RESOLVED
    assert store.get_fact(2) is None
    assert store.get_fact(3) is not None


async def test_expire_memory_is_empty_without_expired_facts(
    settings: Settings, tmp_path: Path
) -> None:
    provider = FakeLlmProvider([completion(report_json())])
    engine, _ = make_engine(settings, tmp_path, provider)
    assert engine.expire_memory() == []

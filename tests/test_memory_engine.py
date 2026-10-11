import logging
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from open_endurance_coach.config import Settings
from open_endurance_coach.engine import coach as coach_module
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport
from open_endurance_coach.schemas.memory import MemoryCategory, MemoryOrigin, MemoryStatus

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


async def test_update_merges_only_the_provided_fields(settings: Settings, tmp_path: Path) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(memory=[memory_op(expires_on="2099-01-01")])),
            completion(
                report_json(
                    memory=[{"action": "update", "id": 1, "text": "Pas de vélo jusqu'au 21/10"}]
                )
            ),
            completion(report_json(memory=[{"action": "update", "id": 1, "expires_on": None}])),
        ]
    )
    engine, store = make_engine(settings, tmp_path, provider)

    await engine.analyze("first", memory_notices=[])
    await engine.analyze("second", memory_notices=[])

    fact = store.list_active_facts()[0]
    assert fact.id == 1
    assert fact.text == "Pas de vélo jusqu'au 21/10"
    assert fact.expires_on == date(2099, 1, 1)

    await engine.analyze("third", memory_notices=[])

    fact = store.list_active_facts()[0]
    assert fact.text == "Pas de vélo jusqu'au 21/10"
    assert fact.expires_on is None


async def test_an_update_without_a_change_is_skipped(
    settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(memory=[memory_op()])),
            completion(report_json(memory=[{"action": "update", "id": 1}])),
        ]
    )
    engine, store = make_engine(settings, tmp_path, provider)
    notices: list[str] = []

    await engine.analyze("first", memory_notices=[])
    with caplog.at_level(logging.WARNING):
        await engine.analyze("second", memory_notices=notices)

    assert notices == []
    assert "update carries no change" in caplog.text
    assert store.list_active_facts()[0].text == "Pas de vélo jusqu'au 20/10"


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


def remember(store: Any, category: MemoryCategory, text: str, **overrides: Any) -> int:
    return store.remember_fact(category, MemoryOrigin.DECLARED, text, **overrides)


async def test_memory_is_injected_into_the_next_prompt(settings: Settings, tmp_path: Path) -> None:
    provider = FakeLlmProvider(
        [
            completion(
                report_json(
                    memory=[memory_op(category="injury_niggle", text="Tendinite genou gauche")]
                )
            ),
            completion(report_json()),
        ]
    )
    engine, _ = make_engine(settings, tmp_path, provider)

    await engine.analyze("first", memory_notices=[])
    await engine.analyze("second", memory_notices=[])

    second_prompt = provider.calls[1]["messages"][1].content
    assert "coach_memory" in second_prompt
    assert "Tendinite genou gauche" in second_prompt


async def test_memory_injection_orders_by_priority_and_caps(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = make_engine(settings, tmp_path, FakeLlmProvider())
    for index in range(3):
        remember(store, MemoryCategory.OBSERVED_PATTERNS, f"Pattern {index}")
    remember(store, MemoryCategory.HEALTH_MEDICAL, "Asthme d'effort")
    monkeypatch.setattr(coach_module, "MEMORY_MAX_FACTS", 2)

    context = await engine.build_context("status check")

    assert [item.category for item in context.coach_memory] == [
        MemoryCategory.HEALTH_MEDICAL,
        MemoryCategory.OBSERVED_PATTERNS,
    ]
    assert len(context.coach_memory_index) == 2
    assert {entry.category for entry in context.coach_memory_index} == {
        MemoryCategory.OBSERVED_PATTERNS
    }


async def test_a_triggered_fact_ranks_before_the_priority_order(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = make_engine(settings, tmp_path, FakeLlmProvider())
    remember(store, MemoryCategory.HEALTH_MEDICAL, "Asthme d'effort")
    remember(
        store,
        MemoryCategory.OBSERVED_PATTERNS,
        "Perd la tenue de FC dans les côtes",
        trigger_keywords=["hills"],
    )
    monkeypatch.setattr(coach_module, "MEMORY_MAX_FACTS", 1)

    context = await engine.build_context("review my hills session")

    assert [item.category for item in context.coach_memory] == [MemoryCategory.OBSERVED_PATTERNS]


async def test_injected_facts_are_marked_as_surfaced(settings: Settings, tmp_path: Path) -> None:
    engine, store = make_engine(settings, tmp_path, FakeLlmProvider())
    fact_id = remember(store, MemoryCategory.ATHLETE_PROFILE, "42 ans, 10 ans de vélo")

    await engine.build_context("status check")

    fact = store.get_fact(fact_id)
    assert fact is not None
    assert fact.last_surfaced_at is not None


async def test_a_long_fact_never_breaks_the_context_budget(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = make_engine(settings, tmp_path, FakeLlmProvider())
    remember(store, MemoryCategory.ATHLETE_PROFILE, "x" * 20000)
    monkeypatch.setattr(coach_module, "MEMORY_RESERVE_TOKENS", 100000)

    context = await engine.build_context("status check")

    assert context.data_tokens() <= context.max_tokens


async def test_refocus_context_keeps_the_memory(settings: Settings, tmp_path: Path) -> None:
    engine, store = make_engine(settings, tmp_path, FakeLlmProvider())
    remember(store, MemoryCategory.INJURY_NIGGLE, "Tendinite genou gauche")

    context = await engine.build_context("status check")
    refocused = engine.refocus_context(context, "plan next week")

    assert [item.text for item in refocused.coach_memory] == ["Tendinite genou gauche"]


async def test_a_reloaded_report_replays_the_same_memory_ops(
    settings: Settings, tmp_path: Path
) -> None:
    engine, store = make_engine(settings, tmp_path, FakeLlmProvider())
    fact_id = remember(
        store, MemoryCategory.TRAINING_CONSTRAINTS, "Pas de vélo", expires_on=date(2099, 1, 1)
    )
    report = DecisionReport.model_validate(
        {"summary": "Note", "memory": [{"action": "update", "id": fact_id}]}
    )
    proposal_id = store.save_proposal(focus="f", report=report, context=CoachContext(focus="f"))

    reloaded = store.get_proposal(proposal_id)

    assert reloaded is not None
    assert engine.apply_memory_ops(reloaded.report.memory) == []
    fact = store.get_fact(fact_id)
    assert fact is not None
    assert fact.expires_on == date(2099, 1, 1)


async def test_memory_ops_apply_from_the_chat_only_fallback(
    settings: Settings, tmp_path: Path
) -> None:
    bad = completion('{"hallucinated": true}')
    chat = completion(report_json(intent="chat", memory=[memory_op()]))
    provider = FakeLlmProvider([bad, bad, bad, bad, chat])
    engine, store = make_engine(settings, tmp_path, provider)
    notices: list[str] = []

    await engine.analyze("no bike", memory_notices=notices)

    assert notices == ["Remembered: [training_constraints] Pas de vélo jusqu'au 20/10"]
    assert len(store.list_active_facts()) == 1


async def test_memory_injection_when_there_is_no_room(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = make_engine(settings, tmp_path, FakeLlmProvider())
    remember(store, MemoryCategory.ATHLETE_PROFILE, "42 ans")
    monkeypatch.setattr(coach_module, "MEMORY_RESERVE_TOKENS", 0)
    monkeypatch.setattr(coach_module, "MEMORY_INDEX_RESERVE_TOKENS", 0)

    context = await engine.build_context("status check")

    assert context.coach_memory == []
    assert context.coach_memory_index == []


async def test_the_index_is_bounded_by_its_own_allowance(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, store = make_engine(settings, tmp_path, FakeLlmProvider())
    remember(store, MemoryCategory.ATHLETE_PROFILE, "42 ans")
    remember(store, MemoryCategory.ATHLETE_PROFILE, "10 ans de vélo")
    monkeypatch.setattr(coach_module, "MEMORY_MAX_FACTS", 1)
    monkeypatch.setattr(coach_module, "MEMORY_INDEX_RESERVE_TOKENS", 0)

    context = await engine.build_context("status check")

    assert len(context.coach_memory) == 1
    assert context.coach_memory_index == []


async def test_an_update_on_a_missing_id_is_skipped(
    settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(memory=[{"action": "update", "id": 999, "text": "x"}]))]
    )
    engine, _ = make_engine(settings, tmp_path, provider)
    notices: list[str] = []

    with caplog.at_level(logging.WARNING):
        await engine.analyze("first", memory_notices=notices)

    assert notices == []
    assert "memory fact not found: 999" in caplog.text

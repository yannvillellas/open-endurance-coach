from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import (
    MemoryCategory,
    MemoryOrigin,
    MemoryStatus,
)

from .fakes import FakeClock

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
TODAY = NOW.date()


def make_store(tmp_path: Path, clock: FakeClock | None = None) -> CoachStore:
    return CoachStore(tmp_path / "coach.db", clock=clock or FakeClock(NOW))


def test_remember_fact_round_trip(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    fact_id = store.remember_fact(
        MemoryCategory.INJURY_NIGGLE,
        MemoryOrigin.DECLARED,
        "Tendinite genou gauche",
        expires_on=date(2026, 10, 20),
        trigger_keywords=["côtes", "hills"],
        source_message_id=7,
    )

    facts = store.list_active_facts(today=TODAY)
    assert [fact.id for fact in facts] == [fact_id]
    fact = facts[0]
    assert fact.category is MemoryCategory.INJURY_NIGGLE
    assert fact.origin is MemoryOrigin.DECLARED
    assert fact.text == "Tendinite genou gauche"
    assert fact.status is MemoryStatus.ACTIVE
    assert fact.valid_from == TODAY
    assert fact.valid_to is None
    assert fact.expires_on == date(2026, 10, 20)
    assert fact.trigger_keywords == ("côtes", "hills")
    assert fact.trigger_on is None
    assert fact.last_surfaced_at is None
    assert fact.source_message_id == 7
    assert fact.created_at == NOW


def test_update_fact_replaces_in_place(tmp_path: Path) -> None:
    clock = FakeClock(NOW)
    store = make_store(tmp_path, clock)
    fact_id = store.remember_fact(
        MemoryCategory.TRAINING_CONSTRAINTS,
        MemoryOrigin.DECLARED,
        "Pas de vélo (déplacement)",
    )

    clock.now = NOW + timedelta(days=1)
    store.update_fact(fact_id, "Pas de vélo jusqu'au 20/10", expires_on=date(2026, 10, 20))

    facts = store.list_active_facts(today=TODAY + timedelta(days=1))
    assert [fact.id for fact in facts] == [fact_id]
    fact = facts[0]
    assert fact.text == "Pas de vélo jusqu'au 20/10"
    assert fact.expires_on == date(2026, 10, 20)
    assert fact.valid_from == TODAY
    assert fact.created_at == NOW
    assert fact.updated_at == NOW + timedelta(days=1)


def test_update_fact_unknown_id_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        store.update_fact(999, "text")


def test_close_history_category_keeps_a_resolved_fact(tmp_path: Path) -> None:
    clock = FakeClock(NOW)
    store = make_store(tmp_path, clock)
    fact_id = store.remember_fact(
        MemoryCategory.INJURY_NIGGLE, MemoryOrigin.DECLARED, "Douleur genou gauche"
    )

    clock.now = NOW + timedelta(days=5)
    outcome = store.close_fact(fact_id)

    assert outcome == "resolved"
    assert store.list_active_facts(today=TODAY + timedelta(days=5)) == []
    fact = store.get_fact(fact_id)
    assert fact is not None
    assert fact.status is MemoryStatus.RESOLVED
    assert fact.valid_to == TODAY + timedelta(days=5)


def test_close_other_category_deletes_the_fact(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    fact_id = store.remember_fact(
        MemoryCategory.TRAINING_CONSTRAINTS, MemoryOrigin.DECLARED, "Pas de vélo"
    )

    outcome = store.close_fact(fact_id)

    assert outcome == "deleted"
    assert store.get_fact(fact_id) is None


def test_forget_fact_deletes_even_a_history_fact(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    fact_id = store.remember_fact(
        MemoryCategory.HEALTH_MEDICAL, MemoryOrigin.DECLARED, "Asthme d'effort"
    )

    store.forget_fact(fact_id)

    assert store.get_fact(fact_id) is None


def test_list_active_facts_excludes_expired_and_resolved(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    active_id = store.remember_fact(
        MemoryCategory.TRAINING_CONSTRAINTS, MemoryOrigin.DECLARED, "Piscine fermée"
    )
    expired_id = store.remember_fact(
        MemoryCategory.TRAINING_CONSTRAINTS,
        MemoryOrigin.DECLARED,
        "Pas de vélo",
        expires_on=date(2026, 9, 10),
    )
    resolved_id = store.remember_fact(
        MemoryCategory.INJURY_NIGGLE, MemoryOrigin.DECLARED, "Ancienne douleur"
    )
    store.close_fact(resolved_id, today=TODAY)

    facts = store.list_active_facts(today=TODAY)

    assert [fact.id for fact in facts] == [active_id]
    assert store.get_fact(expired_id) is not None


def test_facts_as_of_uses_the_validity_window(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    fact_id = store.remember_fact(
        MemoryCategory.INJURY_NIGGLE,
        MemoryOrigin.DECLARED,
        "Douleur genou gauche",
        valid_from=date(2026, 9, 1),
    )
    store.close_fact(fact_id, today=date(2026, 9, 10))

    assert [fact.id for fact in store.facts_as_of(date(2026, 9, 5))] == [fact_id]
    assert store.facts_as_of(date(2026, 9, 11)) == []
    assert [fact.id for fact in store.facts_as_of(date(2026, 9, 1))] == [fact_id]


def test_expire_facts_applies_the_category_policy(tmp_path: Path) -> None:
    clock = FakeClock(NOW)
    store = make_store(tmp_path, clock)
    injury_id = store.remember_fact(
        MemoryCategory.INJURY_NIGGLE,
        MemoryOrigin.DECLARED,
        "Douleur genou",
        expires_on=date(2026, 9, 30),
    )
    constraint_id = store.remember_fact(
        MemoryCategory.TRAINING_CONSTRAINTS,
        MemoryOrigin.DECLARED,
        "Pas de vélo",
        expires_on=date(2026, 9, 30),
    )
    future_id = store.remember_fact(
        MemoryCategory.TRAINING_CONSTRAINTS,
        MemoryOrigin.DECLARED,
        "Piscine fermée",
        expires_on=date(2026, 12, 1),
    )

    outcomes = store.expire_facts(today=date(2026, 10, 1))

    assert outcomes == [(injury_id, "resolved"), (constraint_id, "deleted")]
    injury = store.get_fact(injury_id)
    assert injury is not None
    assert injury.status is MemoryStatus.RESOLVED
    assert injury.valid_to == date(2026, 9, 30)
    assert store.get_fact(constraint_id) is None
    assert store.get_fact(future_id) is not None


def test_mark_facts_surfaced_records_the_timestamp(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    fact_id = store.remember_fact(MemoryCategory.ATHLETE_PROFILE, MemoryOrigin.DECLARED, "42 ans")

    updated = store.mark_facts_surfaced([fact_id], at=NOW)

    assert updated == 1
    fact = store.get_fact(fact_id)
    assert fact is not None
    assert fact.last_surfaced_at == NOW


def test_delete_all_local_erases_memory(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.remember_fact(MemoryCategory.ATHLETE_PROFILE, MemoryOrigin.DECLARED, "42 ans")

    counts = store.delete_all_local()

    assert counts["coach_memory"] == 1
    assert store.list_active_facts(today=TODAY) == []

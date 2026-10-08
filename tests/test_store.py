import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import (
    CreateWorkout,
    DecisionReport,
    DeleteWorkout,
    UpdateWorkout,
)
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import MessageRole, ProposalStatus

NOW = datetime(2024, 2, 1, 12, 0, 0, tzinfo=UTC)


def make_report() -> DecisionReport:
    return DecisionReport(
        summary="Load stable.",
        findings=["Tempo block hit target."],
        questions=["RPE on Thursday?"],
        mutations=[
            CreateWorkout(action="create", name="Tempo Session", start_date_local=date(2024, 2, 5)),
            UpdateWorkout(action="update", event_id=10001, moving_time=4200),
            DeleteWorkout(action="delete", event_id=10002),
        ],
    )


def make_context(focus: str = "status check") -> CoachContext:
    return CoachContext(focus=focus)


def make_store(tmp_path: Path) -> CoachStore:
    from tests.fakes import FakeClock

    return CoachStore(tmp_path / "coach.db", clock=FakeClock(NOW))


def test_new_store_is_empty(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.list_proposals() == []
    assert store.list_unapplied_proposals() == []
    assert store.unseen_activity_ids(["a", "b"]) == {"a", "b"}


def test_mark_activities_seen_returns_new_count(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.mark_activities_seen(["act-1"]) == 1
    assert store.mark_activities_seen(["act-1"]) == 0
    assert store.is_activity_seen("act-1") is True
    assert store.is_activity_seen("act-2") is False


def test_mark_activities_seen_batches_multiple(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.mark_activities_seen(["act-1", "act-2", "act-1", "act-3"]) == 3
    assert store.is_activity_seen("act-1") is True
    assert store.is_activity_seen("act-2") is True
    assert store.is_activity_seen("act-3") is True
    assert store.is_activity_seen("act-4") is False


def test_mark_activities_seen_handles_empty_input(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.mark_activities_seen([]) == 0


def test_unseen_activity_ids_filters_seen(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.mark_activities_seen(["act-1"])
    assert store.unseen_activity_ids(["act-1", "act-2", "act-3"]) == {"act-2", "act-3"}


def test_unseen_activity_ids_queries_only_candidates(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.mark_activities_seen([f"seen-{index}" for index in range(100)])
    queries: list[str] = []
    store._connection.set_trace_callback(lambda statement: queries.append(statement))
    result = store.unseen_activity_ids(["fx-a", "fx-b", "fx-c"])
    assert result == {"fx-a", "fx-b", "fx-c"}
    assert len(queries) == 1
    assert "WHERE activity_id IN" in queries[0]
    assert "fx-a" in queries[0]


def test_unseen_activity_ids_handles_empty_input(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.unseen_activity_ids([]) == set()


def test_save_and_get_proposal_round_trips(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal_id = store.save_proposal(
        focus="Analyze this week",
        report=make_report(),
        context=make_context(),
        user_feedback="Felt tired",
    )
    proposal = store.get_proposal(proposal_id)
    assert proposal is not None
    assert proposal.id == proposal_id
    assert proposal.created_at == NOW
    assert proposal.status is ProposalStatus.PENDING
    assert proposal.focus == "Analyze this week"
    assert proposal.user_feedback == "Felt tired"
    assert proposal.context.focus == "status check"
    assert proposal.report == make_report()


def test_get_proposal_missing_returns_none(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.get_proposal(404) is None


def test_list_proposals_orders_newest_first(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = store.save_proposal(focus="first", report=make_report(), context=make_context("a"))
    second = store.save_proposal(focus="second", report=make_report(), context=make_context("b"))
    proposals = store.list_proposals()
    assert [proposal.id for proposal in proposals] == [second, first]


def test_list_proposals_filters_by_status(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    pending_id = store.save_proposal(focus="pending", report=make_report(), context=make_context())
    approved_id = store.save_proposal(
        focus="approved", report=make_report(), context=make_context()
    )
    store.approve_proposal(approved_id)
    pending = store.list_proposals(ProposalStatus.PENDING)
    approved = store.list_proposals(ProposalStatus.UNAPPLIED)
    assert [proposal.id for proposal in pending] == [pending_id]
    assert [proposal.id for proposal in approved] == [approved_id]


def test_update_proposal_report_replaces_report_and_feedback(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal_id = store.save_proposal(focus="first", report=make_report(), context=make_context())
    replacement = DecisionReport(summary="Revised.", questions=["Any soreness?"])
    store.update_proposal_report(proposal_id, report=replacement, user_feedback="Legs were heavy")
    proposal = store.get_proposal(proposal_id)
    assert proposal is not None
    assert proposal.report == replacement
    assert proposal.user_feedback == "Legs were heavy"
    assert proposal.status is ProposalStatus.PENDING


def test_update_proposal_report_with_context_persists_context(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal_id = store.save_proposal(focus="first", report=make_report(), context=make_context())
    replacement = DecisionReport(summary="Revised.")
    updated_context = make_context("status check with feedback")
    updated_context = CoachContext(
        focus=updated_context.focus, user_feedback="RPE 8", max_tokens=updated_context.max_tokens
    )
    store.update_proposal_report(
        proposal_id, report=replacement, user_feedback="RPE 8", context=updated_context
    )
    proposal = store.get_proposal(proposal_id)
    assert proposal is not None
    assert proposal.context.user_feedback == "RPE 8"
    assert proposal.context.focus == "status check with feedback"


def test_update_proposal_report_missing_proposal_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        store.update_proposal_report(404, report=make_report(), user_feedback=None)


def test_update_proposal_report_non_pending_proposal_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal_id = store.save_proposal(focus="first", report=make_report(), context=make_context())
    store.approve_proposal(proposal_id)
    with pytest.raises(ValueError, match="pending"):
        store.update_proposal_report(proposal_id, report=make_report(), user_feedback=None)


def test_add_message_records_rows(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = store.add_message(MessageRole.USER, "RPE was 7")
    second = store.add_message(MessageRole.ASSISTANT, "Noted.", report=make_report())
    messages = store.list_messages()
    assert [item.id for item in messages] == [first, second]
    assert [item.content for item in messages] == ["RPE was 7", "Noted."]
    assert [item.role for item in messages] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert messages[0].report is None
    assert messages[1].report == make_report()
    assert all(item.created_at == NOW for item in messages)


def test_approve_proposal_creates_proposal_and_flips_status(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal_id = store.save_proposal(focus="first", report=make_report(), context=make_context())
    proposal = store.approve_proposal(proposal_id)
    assert proposal.id == proposal_id
    assert proposal.decided_at == NOW
    assert proposal.approved_report == make_report()
    stored = store.get_proposal(proposal_id)
    assert stored is not None
    assert stored.status is ProposalStatus.UNAPPLIED
    proposals = store.list_unapplied_proposals()
    assert len(proposals) == 1
    assert proposals[0].id == stored.id


def test_approve_proposal_records_the_approved_report(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal_id = store.save_proposal(focus="first", report=make_report(), context=make_context())
    subset = make_report().model_copy(update={"mutations": make_report().mutations[2:]})
    proposal = store.approve_proposal(proposal_id, report=subset)
    assert proposal.approved_report == subset
    stored = store.get_proposal(proposal.id)
    assert stored is not None
    assert stored.approved_report == subset
    stored = store.get_proposal(proposal_id)
    assert stored is not None
    assert stored.status is ProposalStatus.UNAPPLIED
    assert stored.report == make_report()


def test_approve_missing_proposal_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        store.approve_proposal(404)


def test_approve_twice_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal_id = store.save_proposal(focus="first", report=make_report(), context=make_context())
    store.approve_proposal(proposal_id)
    with pytest.raises(ValueError, match="pending"):
        store.approve_proposal(proposal_id)
    assert len(store.list_unapplied_proposals()) == 1


def test_approve_proposal_is_atomic_when_the_update_fails(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal_id = store.save_proposal(focus="first", report=make_report(), context=make_context())
    store._connection.execute(
        "CREATE TRIGGER refuse_proposal BEFORE UPDATE ON proposals"
        " BEGIN SELECT RAISE(ABORT, 'disk full'); END;"
    )
    store._connection.commit()
    with pytest.raises(sqlite3.IntegrityError):
        store.approve_proposal(proposal_id)
    proposal = store.get_proposal(proposal_id)
    assert proposal is not None
    assert proposal.status is ProposalStatus.PENDING
    assert proposal.approved_report is None
    assert store.list_unapplied_proposals() == []
    store._connection.execute("DROP TRIGGER refuse_proposal")
    store._connection.commit()
    proposal = store.approve_proposal(proposal_id)
    assert proposal.id == proposal_id
    reopened = store.get_proposal(proposal_id)
    assert reopened is not None
    assert reopened.status is ProposalStatus.UNAPPLIED
    assert len(store.list_unapplied_proposals()) == 1


def test_store_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "coach.db"
    from tests.fakes import FakeClock

    store = CoachStore(path, clock=FakeClock(NOW))
    proposal_id = store.save_proposal(
        focus="first", report=make_report(), context=make_context(), user_feedback="tired"
    )
    store.mark_activities_seen(["act-1"])
    store.close()
    reopened = CoachStore(path, clock=FakeClock(NOW))
    proposal = reopened.get_proposal(proposal_id)
    assert proposal is not None
    assert proposal.report == make_report()
    assert proposal.user_feedback == "tired"
    assert reopened.is_activity_seen("act-1") is True


def test_store_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "coach.db"
    store = CoachStore(path)
    assert path.exists()
    store.close()


def approve_proposal(store: CoachStore) -> int:
    proposal_id = store.save_proposal(focus="first", report=make_report(), context=make_context())
    return store.approve_proposal(proposal_id).id


def test_store_creates_the_fresh_proposals_schema(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "coach.db"
    store = CoachStore(path)
    proposal_id = store.save_proposal(focus="first", report=make_report(), context=make_context())
    proposal = store.get_proposal(proposal_id)
    assert proposal is not None
    assert proposal.approved_report is None
    store.close()
    connection = sqlite3.connect(path)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(proposals)").fetchall()}
    connection.close()
    assert {"approved_json", "decided_at"} <= columns


def test_recent_messages_respects_rows_without_a_report(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_message(MessageRole.USER, "old row")
    old = store.recent_messages(1)[0]
    assert old.content == "old row"
    assert old.report is None
    store.add_message(MessageRole.ASSISTANT, "new row", report=DecisionReport(summary="Answer."))
    new = store.recent_messages(1)[0]
    assert new.report is not None
    assert new.report.summary == "Answer."


def test_recent_messages_empty_store(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.recent_messages(10) == []


def test_recent_messages_zero_or_negative_limit_returns_empty(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_message(MessageRole.USER, "RPE was 7")
    assert store.recent_messages(0) == []
    assert store.recent_messages(-3) == []


def test_recent_messages_carry_their_own_report(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_message(MessageRole.ASSISTANT, "RPE was 7", report=make_report())
    rows = store.recent_messages(10)
    assert len(rows) == 1
    assert rows[0].content == "RPE was 7"
    assert rows[0].report == make_report()


def test_recent_messages_orders_newest_first(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_message(MessageRole.USER, "one")
    store.add_message(MessageRole.USER, "two")
    store.add_message(MessageRole.USER, "three")
    rows = store.recent_messages(10)
    assert [row.content for row in rows] == ["three", "two", "one"]


def test_recent_messages_limit_caps_the_window(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    for index in range(5):
        store.add_message(MessageRole.USER, f"note {index}")
    rows = store.recent_messages(2)
    assert [row.content for row in rows] == ["note 4", "note 3"]


def test_recent_messages_cutoff_filters_old_rows(tmp_path: Path) -> None:
    from tests.fakes import FakeClock

    clock = FakeClock(NOW)
    store = CoachStore(tmp_path / "coach.db", clock=clock)
    store.add_message(MessageRole.USER, "old row")
    clock.now = NOW + timedelta(days=10)
    store.add_message(MessageRole.USER, "recent row")
    recent = store.recent_messages(10, max_age_days=5)
    assert [row.content for row in recent] == ["recent row"]
    assert all(row.created_at == NOW + timedelta(days=10) for row in recent)
    unfiltered = store.recent_messages(10)
    assert [row.content for row in unfiltered] == ["recent row", "old row"]


def test_prune_before_deletes_old_rows_and_keeps_recent(tmp_path: Path) -> None:
    from tests.fakes import FakeClock

    clock = FakeClock(NOW)
    store = CoachStore(tmp_path / "coach.db", clock=clock)
    old_proposal = store.save_proposal(focus="old", report=make_report(), context=make_context())
    store.add_message(MessageRole.USER, "old feedback")
    store.mark_activities_seen(["fx-old"])
    store.approve_proposal(old_proposal)

    clock.now = NOW + timedelta(days=200)
    store.save_proposal(focus="recent", report=make_report(), context=make_context())
    store.add_message(MessageRole.USER, "recent feedback")
    store.mark_activities_seen(["fx-recent"])

    counts = store.prune_before(NOW + timedelta(days=100))

    assert counts == {"messages": 1, "proposals": 1, "seen_activities": 1}
    assert [proposal.focus for proposal in store.list_proposals()] == ["recent"]
    assert [row.content for row in store.recent_messages(10)] == ["recent feedback"]
    assert store.list_unapplied_proposals() == []
    assert store.unseen_activity_ids(["fx-old", "fx-recent"]) == {"fx-old"}


def test_prune_before_keeps_messages_newer_than_the_cutoff(tmp_path: Path) -> None:
    from tests.fakes import FakeClock

    clock = FakeClock(NOW)
    store = CoachStore(tmp_path / "coach.db", clock=clock)
    proposal_id = store.save_proposal(focus="old", report=make_report(), context=make_context())

    clock.now = NOW + timedelta(days=200)
    store.add_message(MessageRole.USER, "late feedback")
    store.approve_proposal(proposal_id)

    counts = store.prune_before(NOW + timedelta(days=100))

    assert counts == {"messages": 0, "proposals": 1, "seen_activities": 0}
    assert [row.content for row in store.list_messages()] == ["late feedback"]
    assert store.list_proposals() == []
    assert store.list_unapplied_proposals() == []


def test_database_file_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "coach.db"
    CoachStore(path)
    assert (path.stat().st_mode & 0o777) == 0o600


def test_unseen_activity_ids_handles_large_batches(tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    seen = [f"fx-{index}" for index in range(1000)]
    store.mark_activities_seen(seen)
    unseen = store.unseen_activity_ids([*seen, "fx-new-1", "fx-new-2"])
    assert unseen == {"fx-new-1", "fx-new-2"}


def test_reject_proposal_deletes_it_and_keeps_messages(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal_id = store.save_proposal(focus="first", report=make_report(), context=make_context())
    store.add_message(MessageRole.USER, "legs heavy")
    store.reject_proposal(proposal_id)
    assert store.get_proposal(proposal_id) is None
    assert store.list_proposals() == []
    assert [row.content for row in store.list_messages()] == ["legs heavy"]
    with pytest.raises(ValueError, match="not found"):
        store.reject_proposal(proposal_id)


def test_prune_does_not_vacuum_for_a_single_row(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    proposal_id = store.save_proposal(focus="old", report=make_report(), context=make_context())
    old_stamp = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    store._connection.execute(
        "UPDATE proposals SET created_at = ? WHERE id = ?", (old_stamp, proposal_id)
    )
    store._connection.commit()
    traced: list[str] = []
    store._connection.set_trace_callback(lambda statement: traced.append(statement.upper()))
    counts = store.prune_before(datetime.now(UTC) - timedelta(days=100))
    assert counts["proposals"] == 1
    assert not any(statement.startswith("VACUUM") for statement in traced)


def test_prune_messages_deletes_messages_only(tmp_path: Path) -> None:
    from tests.fakes import FakeClock

    clock = FakeClock(NOW)
    store = CoachStore(tmp_path / "coach.db", clock=clock)
    proposal_id = store.save_proposal(focus="old", report=make_report(), context=make_context())
    store.add_message(MessageRole.USER, "old message")
    store.mark_activities_seen(["fx-old"])

    clock.now = NOW + timedelta(days=200)
    store.add_message(MessageRole.USER, "recent message")
    store.mark_activities_seen(["fx-recent"])

    removed = store.prune_messages(NOW + timedelta(days=100))

    assert removed == 1
    assert [row.content for row in store.list_messages()] == ["recent message"]
    assert store.get_proposal(proposal_id) is not None
    assert store.is_activity_seen("fx-old")
    assert store.is_activity_seen("fx-recent")


def test_delete_all_local_wipes_messages_proposals_and_seen(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.save_proposal(focus="old", report=make_report(), context=make_context())
    store.add_message(MessageRole.USER, "hello")
    store.mark_activities_seen(["fx-1"])

    counts = store.delete_all_local()

    assert counts == {"messages": 1, "proposals": 1, "seen_activities": 1}
    assert store.list_messages() == []
    assert store.list_proposals() == []
    assert store.is_activity_seen("fx-1") is False

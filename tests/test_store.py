from datetime import UTC, date, datetime
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
from open_endurance_coach.store.records import DraftStatus

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
    assert store.list_drafts() == []
    assert store.list_decisions() == []
    assert store.unseen_activity_ids(["a", "b"]) == {"a", "b"}


def test_mark_activity_seen_returns_true_once(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.mark_activity_seen("act-1") is True
    assert store.mark_activity_seen("act-1") is False
    assert store.is_activity_seen("act-1") is True
    assert store.is_activity_seen("act-2") is False


def test_unseen_activity_ids_filters_seen(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.mark_activity_seen("act-1")
    assert store.unseen_activity_ids(["act-1", "act-2", "act-3"]) == {"act-2", "act-3"}


def test_unseen_activity_ids_handles_empty_input(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.unseen_activity_ids([]) == set()


def test_save_and_get_draft_round_trips(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft_id = store.save_draft(
        focus="Analyze this week",
        report=make_report(),
        context=make_context(),
        user_feedback="Felt tired",
    )
    draft = store.get_draft(draft_id)
    assert draft is not None
    assert draft.id == draft_id
    assert draft.created_at == NOW
    assert draft.status is DraftStatus.PENDING
    assert draft.focus == "Analyze this week"
    assert draft.user_feedback == "Felt tired"
    assert draft.context.focus == "status check"
    assert draft.report == make_report()


def test_get_draft_missing_returns_none(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    assert store.get_draft(404) is None


def test_list_drafts_orders_newest_first(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = store.save_draft(focus="first", report=make_report(), context=make_context("a"))
    second = store.save_draft(focus="second", report=make_report(), context=make_context("b"))
    drafts = store.list_drafts()
    assert [draft.id for draft in drafts] == [second, first]


def test_list_drafts_filters_by_status(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    pending_id = store.save_draft(focus="pending", report=make_report(), context=make_context())
    approved_id = store.save_draft(focus="approved", report=make_report(), context=make_context())
    store.approve_draft(approved_id)
    pending = store.list_drafts(DraftStatus.PENDING)
    approved = store.list_drafts(DraftStatus.APPROVED)
    assert [draft.id for draft in pending] == [pending_id]
    assert [draft.id for draft in approved] == [approved_id]


def test_update_draft_report_replaces_report_and_feedback(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft_id = store.save_draft(focus="first", report=make_report(), context=make_context())
    replacement = DecisionReport(summary="Revised.", questions=["Any soreness?"])
    store.update_draft_report(draft_id, report=replacement, user_feedback="Legs were heavy")
    draft = store.get_draft(draft_id)
    assert draft is not None
    assert draft.report == replacement
    assert draft.user_feedback == "Legs were heavy"
    assert draft.status is DraftStatus.PENDING


def test_update_draft_report_missing_draft_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        store.update_draft_report(404, report=make_report(), user_feedback=None)


def test_update_draft_report_non_pending_draft_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft_id = store.save_draft(focus="first", report=make_report(), context=make_context())
    store.approve_draft(draft_id)
    with pytest.raises(ValueError, match="pending"):
        store.update_draft_report(draft_id, report=make_report(), user_feedback=None)


def test_add_feedback_records_rows(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft_id = store.save_draft(focus="first", report=make_report(), context=make_context())
    first = store.add_feedback(draft_id, "RPE was 7")
    second = store.add_feedback(draft_id, "Slept poorly")
    feedback = store.list_feedback(draft_id)
    assert [item.id for item in feedback] == [first, second]
    assert [item.content for item in feedback] == ["RPE was 7", "Slept poorly"]
    assert all(item.draft_id == draft_id for item in feedback)
    assert all(item.created_at == NOW for item in feedback)


def test_add_feedback_missing_draft_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        store.add_feedback(404, "RPE was 7")


def test_approve_draft_creates_decision_and_flips_status(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft_id = store.save_draft(focus="first", report=make_report(), context=make_context())
    decision = store.approve_draft(draft_id)
    assert decision.draft_id == draft_id
    assert decision.decided_at == NOW
    assert decision.report == make_report()
    draft = store.get_draft(draft_id)
    assert draft is not None
    assert draft.status is DraftStatus.APPROVED
    decisions = store.list_decisions()
    assert len(decisions) == 1
    assert decisions[0].id == decision.id


def test_approve_missing_draft_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        store.approve_draft(404)


def test_approve_twice_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft_id = store.save_draft(focus="first", report=make_report(), context=make_context())
    store.approve_draft(draft_id)
    with pytest.raises(ValueError, match="pending"):
        store.approve_draft(draft_id)
    assert len(store.list_decisions()) == 1


def test_reject_draft_flips_status(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft_id = store.save_draft(focus="first", report=make_report(), context=make_context())
    store.reject_draft(draft_id)
    draft = store.get_draft(draft_id)
    assert draft is not None
    assert draft.status is DraftStatus.REJECTED
    assert store.list_decisions() == []


def test_reject_missing_draft_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        store.reject_draft(404)


def test_reject_approved_draft_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    draft_id = store.save_draft(focus="first", report=make_report(), context=make_context())
    store.approve_draft(draft_id)
    with pytest.raises(ValueError, match="pending"):
        store.reject_draft(draft_id)


def test_store_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "coach.db"
    from tests.fakes import FakeClock

    store = CoachStore(path, clock=FakeClock(NOW))
    draft_id = store.save_draft(
        focus="first", report=make_report(), context=make_context(), user_feedback="tired"
    )
    store.mark_activity_seen("act-1")
    store.close()
    reopened = CoachStore(path, clock=FakeClock(NOW))
    draft = reopened.get_draft(draft_id)
    assert draft is not None
    assert draft.report == make_report()
    assert draft.user_feedback == "tired"
    assert reopened.is_activity_seen("act-1") is True


def test_store_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "coach.db"
    store = CoachStore(path)
    assert path.exists()
    store.close()

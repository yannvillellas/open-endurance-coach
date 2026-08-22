from datetime import date
from pathlib import Path
from typing import Any

import pytest

from open_endurance_coach.clients.llm import LlmClient, LlmError, LlmMessage
from open_endurance_coach.config import Settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import CreateWorkout, DecisionReport, WorkoutMutation
from open_endurance_coach.schemas.intervals import Activity
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import DraftStatus
from open_endurance_coach.writer.calendar import CalendarWriter

from .fakes import (
    FakeCalendarClient,
    FakeIntervalsClient,
    FakeLlmProvider,
    RecordingSleep,
    completion,
    make_activity,
    make_activity_list,
    make_intervals_client,
    report_json,
)

TODAY = date(2024, 2, 1)

CREATE_MUTATION = {
    "action": "create",
    "name": "Tempo Session",
    "start_date_local": "2024-02-05",
    "moving_time": 3600,
}


def make_engine(
    settings: Settings,
    store: CoachStore,
    provider: FakeLlmProvider,
    client: FakeIntervalsClient | None = None,
    writer: CalendarWriter | None = None,
) -> CoachEngine:
    llm = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": provider},
        sleep=RecordingSleep(),
    )
    return CoachEngine(settings, store, client or make_intervals_client(), llm, writer=writer)


def make_activity_model(activity_id: str, day: int, **overrides: Any) -> Activity:
    payload = make_activity(activity_id, day)
    payload.update(overrides)
    return Activity.model_validate(payload)


async def test_analyze_standard_produces_pending_draft(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("Analyze this week", today=TODAY)
    assert draft.status is DraftStatus.PENDING
    assert draft.report.summary == "Load stable."
    assert draft.user_feedback is None
    assert "New activities since last review" in draft.focus
    for activity_id in ("fx-a", "fx-b", "fx-c", "fx-d", "fx-e"):
        assert store.is_activity_seen(activity_id) is True


async def test_analyze_surfaces_only_unseen_activities(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    first_provider = FakeLlmProvider([completion(report_json())])
    first_engine = make_engine(settings, store, first_provider)
    await first_engine.analyze("Analyze this week", today=TODAY)

    second_client = make_intervals_client(
        activities=[*make_activity_list(), make_activity("fx-z", 21, name="Evening Ride")]
    )
    second_provider = FakeLlmProvider([completion(report_json("Week reviewed."))])
    second_engine = make_engine(settings, store, second_provider, client=second_client)
    second_draft = await second_engine.analyze("Analyze this week", today=TODAY)
    assert "New activities since last review: Evening Ride (2024-01-21)" in second_draft.focus
    listing = second_draft.focus.split("New activities since last review")[1]
    assert "Synthetic Workout" not in listing
    assert store.is_activity_seen("fx-z") is True


async def test_analyze_marks_seen_only_after_success(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(""), completion(""), completion("")])
    engine = make_engine(settings, store, provider)
    with pytest.raises(LlmError, match="failed after 3 attempts"):
        await engine.analyze("status check", today=TODAY)
    assert store.list_drafts() == []
    assert store.unseen_activity_ids(["fx-a", "fx-b"]) == {"fx-a", "fx-b"}


async def test_analyze_retries_on_schema_invalid_response(
    settings: Settings, tmp_path: Path
) -> None:
    provider = FakeLlmProvider([completion('{"hallucinated": true}'), completion(report_json())])
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), provider)
    draft = await engine.analyze("status check", today=TODAY)
    assert draft.report.summary == "Load stable."
    assert len(provider.calls) == 2


async def test_analyze_schema_invalid_exhausts_attempts(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion('{"hallucinated": true}')] * 3)
    engine = make_engine(settings, store, provider)
    with pytest.raises(LlmError, match="failed after 3 attempts"):
        await engine.analyze("status check", today=TODAY)
    assert store.list_drafts() == []


async def test_analyze_uses_deep_extractor_for_deep_focus(
    settings: Settings, tmp_path: Path
) -> None:
    client = make_intervals_client()
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), provider, client=client)
    draft = await engine.analyze(
        "how did my heart rate improve on hills in the last 3 months", today=TODAY
    )
    assert ("detail", "fx-a") in client.calls
    assert ("activities", "2023-11-03", "2024-02-02") in client.calls
    assert "New activities since last review" in draft.focus
    assert draft.context.activity_detail is not None
    assert draft.context.activity_detail.id == "fx-a"


async def test_analyze_injects_user_feedback(settings: Settings, tmp_path: Path) -> None:
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), provider)
    draft = await engine.analyze("status check", user_feedback="Legs heavy", today=TODAY)
    assert draft.user_feedback == "Legs heavy"
    assert "Legs heavy" in provider.calls[0]["messages"][1].content


async def test_review_solicits_missing_rpe(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    context = CoachContext(
        focus="status check",
        recent_activities=[
            make_activity_model("fx-a", 20),
            make_activity_model("fx-b", 18, icu_rpe=7.0),
        ],
    )
    draft_id = store.save_draft(
        focus="status check", report=DecisionReport(summary="ok"), context=context
    )
    engine = make_engine(settings, store, FakeLlmProvider())
    view = engine.review(draft_id)
    assert view.draft.id == draft_id
    assert view.requested_feedback == [
        "RPE missing for Synthetic Workout on 2024-01-20: how hard did it feel (1-10)?",
        "Fueling: carb intake and hydration for the sessions above?",
    ]


async def test_review_no_solicitations_when_rpe_logged(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    context = CoachContext(
        focus="status check",
        recent_activities=[
            make_activity_model("fx-a", 20, icu_rpe=7.0),
            make_activity_model("fx-b", 18, perceived_exertion=6.0),
        ],
    )
    draft_id = store.save_draft(
        focus="status check", report=DecisionReport(summary="ok"), context=context
    )
    engine = make_engine(settings, store, FakeLlmProvider())
    assert engine.review(draft_id).requested_feedback == []


async def test_review_missing_draft_raises(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(ValueError, match="not found"):
        engine.review(404)


async def test_submit_feedback_updates_draft_and_injects_feedback(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider(
        [completion(report_json()), completion(report_json("Revised after feedback."))]
    )
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check", today=TODAY)
    updated = await engine.submit_feedback(draft.id, "Legs heavy, RPE 8")
    assert updated.id == draft.id
    assert updated.report.summary == "Revised after feedback."
    assert updated.user_feedback == "Legs heavy, RPE 8"
    assert updated.status is DraftStatus.PENDING
    assert "Legs heavy, RPE 8" in provider.calls[1]["messages"][1].content
    assert [item.content for item in store.list_feedback(draft.id)] == ["Legs heavy, RPE 8"]


async def test_submit_feedback_persists_feedback_context(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider(
        [completion(report_json()), completion(report_json("Revised after feedback."))]
    )
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check", today=TODAY)
    await engine.submit_feedback(draft.id, "Legs heavy, RPE 8")
    stored = store.get_draft(draft.id)
    assert stored is not None
    assert stored.context.user_feedback == "Legs heavy, RPE 8"


async def test_submit_feedback_over_budget_raises_before_llm(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    context = CoachContext(focus="status check", max_tokens=10)
    draft_id = store.save_draft(
        focus="status check", report=DecisionReport(summary="ok"), context=context
    )
    provider = FakeLlmProvider([completion(report_json("Should not run."))])
    engine = make_engine(settings, store, provider)
    with pytest.raises(ValueError, match="token budget"):
        await engine.submit_feedback(
            draft_id, "A very long feedback text that overflows the budget"
        )
    assert provider.calls == []
    assert store.list_feedback(draft_id) == []
    stored = store.get_draft(draft_id)
    assert stored is not None
    assert stored.context.user_feedback is None


async def test_submit_feedback_falls_back_without_current_proposal_on_budget_overflow(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    big_report = DecisionReport(summary="x" * 400)
    draft_id = store.save_draft(
        focus="f", report=big_report, context=CoachContext(focus="f", max_tokens=100)
    )
    provider = FakeLlmProvider([completion(report_json("Revised.", mutations=[CREATE_MUTATION]))])
    engine = make_engine(settings, store, provider)
    updated = await engine.submit_feedback(draft_id, "make it easier")
    assert [row.content for row in store.list_feedback(draft_id)] == ["make it easier"]
    assert updated.context.current_proposal is None
    assert updated.context.user_feedback == "make it easier"
    assert updated.report.summary == "Revised."


async def test_surface_unseen_falls_back_when_listing_overflows_budget(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    context = CoachContext(
        focus="status check",
        recent_activities=[make_activity_model("fx-a", 20)],
        max_tokens=75,
    )
    engine = make_engine(settings, store, FakeLlmProvider())
    surfaced = engine._surface_unseen(context)
    assert surfaced.focus == "status check"
    assert "New activities since last review" not in surfaced.focus


async def test_submit_feedback_non_pending_raises(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check", today=TODAY)
    engine.approve(draft.id)
    with pytest.raises(ValueError, match="pending"):
        await engine.submit_feedback(draft.id, "too late")


async def test_submit_feedback_missing_draft_raises(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(ValueError, match="not found"):
        await engine.submit_feedback(404, "feedback")


async def test_approve_records_decision(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check", today=TODAY)
    decision = engine.approve(draft.id)
    mutation = decision.report.mutations[0]
    assert isinstance(mutation, CreateWorkout)
    assert mutation.name == "Tempo Session"
    approved = store.get_draft(draft.id)
    assert approved is not None
    assert approved.status is DraftStatus.APPROVED
    assert store.list_decisions() == [decision]


async def test_approve_with_override_mutations(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check", today=TODAY)
    override: list[WorkoutMutation] = [
        CreateWorkout(action="create", name="Custom Session", start_date_local=date(2024, 2, 6))
    ]
    decision = engine.approve(draft.id, mutations=override)
    assert decision.report.mutations == override
    assert decision.report.summary == "Mutations overridden by the athlete."
    assert decision.report.findings == []
    assert decision.report.questions == []


async def test_approve_missing_draft_raises(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(ValueError, match="not found"):
        engine.approve(404)


async def test_reject_flips_status(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check", today=TODAY)
    engine.reject(draft.id)
    rejected = store.get_draft(draft.id)
    assert rejected is not None
    assert rejected.status is DraftStatus.REJECTED
    assert store.list_decisions() == []


async def test_pending_drafts_lists_only_pending(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    first = store.save_draft(
        focus="first", report=DecisionReport(summary="ok"), context=CoachContext(focus="first")
    )
    second = store.save_draft(
        focus="second", report=DecisionReport(summary="ok"), context=CoachContext(focus="second")
    )
    store.approve_draft(first)
    assert [draft.id for draft in engine.pending_drafts()] == [second]


async def test_apply_without_writer_raises(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(RuntimeError, match="writer"):
        await engine.apply()


def applied_decision(store: CoachStore, decision_id: int) -> Any:
    decision = store.get_decision(decision_id)
    assert decision is not None
    return decision


async def test_apply_applies_unapplied_decisions_and_marks_applied(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check", today=TODAY)
    engine.approve(draft.id)
    calendar = FakeCalendarClient()
    writer_engine = make_engine(settings, store, provider, writer=CalendarWriter(calendar))
    report = await writer_engine.apply()
    assert len(report.decisions) == 1
    assert report.decisions[0].decision_id == 1
    assert report.decisions[0].outcomes[0].target == "created"
    assert len(calendar.created) == 1
    assert store.get_decision(1) is not None
    assert applied_decision(store, 1).applied_at is not None
    assert store.list_unapplied_decisions() == []


async def test_apply_dry_run_writes_nothing_and_leaves_unapplied(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check", today=TODAY)
    engine.approve(draft.id)
    calendar = FakeCalendarClient()
    writer_engine = make_engine(settings, store, provider, writer=CalendarWriter(calendar))
    report = await writer_engine.apply(dry_run=True)
    assert report.decisions[0].outcomes[0].target == "created"
    assert calendar.created == []
    assert applied_decision(store, 1).applied_at is None
    assert len(store.list_unapplied_decisions()) == 1


async def test_apply_specific_decision_only(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json()), completion(report_json())])
    engine = make_engine(settings, store, provider)
    first = await engine.analyze("status check", today=TODAY)
    second = await engine.analyze("status check", today=TODAY)
    engine.approve(first.id)
    engine.approve(second.id)
    calendar = FakeCalendarClient()
    writer_engine = make_engine(settings, store, provider, writer=CalendarWriter(calendar))
    report = await writer_engine.apply(decision_id=second.id)
    assert [item.decision_id for item in report.decisions] == [second.id]
    assert applied_decision(store, first.id).applied_at is None
    assert applied_decision(store, second.id).applied_at is not None


async def test_apply_already_applied_decision_raises(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check", today=TODAY)
    engine.approve(draft.id)
    writer_engine = make_engine(
        settings, store, provider, writer=CalendarWriter(FakeCalendarClient())
    )
    await writer_engine.apply()
    with pytest.raises(ValueError, match="already applied"):
        await writer_engine.apply(decision_id=1)


async def test_apply_marks_empty_decision_applied(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check", today=TODAY)
    engine.approve(draft.id)
    calendar = FakeCalendarClient()
    writer_engine = make_engine(settings, store, provider, writer=CalendarWriter(calendar))
    report = await writer_engine.apply()
    assert report.decisions[0].outcomes == []
    assert calendar.created == []
    assert applied_decision(store, 1).applied_at is not None


async def test_converse_reuses_context_without_extraction(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    client = FakeIntervalsClient([], [], [], [])
    provider = FakeLlmProvider([completion("Keep load stable.")])
    engine = make_engine(settings, store, provider, client=client)
    context = CoachContext(focus="how was my week", today=TODAY)
    reply = await engine.converse("what do you think?", context=context, today=TODAY)
    assert reply == "Keep load stable."
    assert client.calls == []
    assert store.list_drafts() == []
    recorded = provider.calls[0]
    assert recorded["json_mode"] is False
    assert recorded["thinking"] is True
    messages = recorded["messages"]
    assert [message.role for message in messages] == ["system", "user", "user"]
    assert messages[-1].content == "what do you think?"
    assert "how was my week" in messages[1].content


async def test_converse_extracts_and_surfaces_unseen_without_marking_seen(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion("Trend looks fine.")])
    engine = make_engine(settings, store, provider)
    reply = await engine.converse("how was my week?", today=TODAY)
    assert reply == "Trend looks fine."
    messages = provider.calls[0]["messages"]
    assert "New activities since last review" in messages[1].content
    assert store.is_activity_seen("fx-a") is False
    assert store.list_drafts() == []


async def test_converse_deep_query_uses_deep_extractor(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    client = make_intervals_client()
    provider = FakeLlmProvider([completion("Hills improving.")])
    engine = make_engine(settings, store, provider, client=client)
    reply = await engine.converse(
        "how much did my heart rate improve on hilly sections over the last 3 months",
        today=TODAY,
    )
    assert reply == "Hills improving."
    assert any(call[0] == "detail" for call in client.calls)


async def test_converse_includes_history_between_data_and_turn(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion("Good point.")])
    engine = make_engine(settings, store, provider)
    history = [
        LlmMessage(role="user", content="past question"),
        LlmMessage(role="assistant", content="past answer"),
    ]
    await engine.converse("follow up", context=CoachContext(focus="f"), history=history)
    messages = provider.calls[0]["messages"]
    assert messages[2] == history[0]
    assert messages[3] == history[1]
    assert messages[4].content == "follow up"


async def test_converse_today_anchor_reaches_prompt(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion("ok")])
    engine = make_engine(settings, store, provider)
    await engine.converse("hi", context=CoachContext(focus="f", today=TODAY))
    messages = provider.calls[0]["messages"]
    assert "Today's date (athlete local): 2024-02-01" in messages[1].content


async def test_converse_empty_content_raises_without_writes(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion("")])
    engine = make_engine(settings, store, provider)
    with pytest.raises(LlmError, match="empty content"):
        await engine.converse("hi", context=CoachContext(focus="f"))
    assert store.list_drafts() == []


async def test_recent_history_reads_feedback_from_store(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    draft_id = store.save_draft(
        focus="f", report=DecisionReport(summary="ok"), context=CoachContext(focus="f")
    )
    store.add_feedback(draft_id, "legs heavy")
    rows = engine.recent_history(10)
    assert len(rows) == 1
    assert rows[0].feedback.content == "legs heavy"
    assert rows[0].report.summary == "ok"
    assert engine.recent_history(0) == []


async def test_build_context_surfaces_unseen_without_marking(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    context = await engine.build_context("how was my week?", today=TODAY)
    assert "New activities since last review" in context.focus
    assert store.is_activity_seen("fx-a") is False
    assert store.list_drafts() == []

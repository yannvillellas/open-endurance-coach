import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.clients.llm import LlmClient, LlmError, LlmMessage
from open_endurance_coach.config import Settings
from open_endurance_coach.engine.coach import (
    MAX_PLANNING_DAYS,
    CoachEngine,
    PlaceholderMutationError,
    StaleDecisionError,
    _validate_report,
)
from open_endurance_coach.prompts.prompts import system_prompt
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import CreateWorkout, DecisionReport
from open_endurance_coach.schemas.intervals import Activity
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import DraftStatus
from open_endurance_coach.tokens import INPUT_TOKEN_CEILING, estimate_text_tokens
from open_endurance_coach.writer.calendar import CalendarWriter

from .fakes import (
    FakeCalendarClient,
    FakeClock,
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


def _near_future(days: int = 30) -> str:
    return (datetime.now(ZoneInfo("Europe/Paris")).date() + timedelta(days=days)).isoformat()


CREATE_MUTATION = {
    "action": "create",
    "name": "Tempo Session",
    "start_date_local": _near_future(),
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


async def test_submit_feedback_trims_an_over_budget_context(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    feedback = "A very long feedback text that must fit the context budget"
    probe = CoachContext(
        focus="status check",
        today=datetime.now(ZoneInfo(settings.app_timezone)).date(),
    )
    context = probe.model_copy(
        update={
            "max_tokens": probe.estimated_tokens() + estimate_text_tokens(feedback) + 5,
            "recent_activities": [],
        }
    )
    draft_id = store.save_draft(
        focus="status check", report=DecisionReport(summary="ok"), context=context
    )
    provider = FakeLlmProvider([completion(report_json("Revised."))])
    engine = make_engine(settings, store, provider)
    updated = await engine.submit_feedback(draft_id, feedback)
    assert updated.report.summary == "Revised."
    assert len(provider.calls) == 1
    prompt = provider.calls[0]["messages"][1].content
    assert feedback in prompt
    assert updated.context.estimated_tokens() <= updated.context.max_tokens


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
    probe = CoachContext(focus="status check", recent_activities=[make_activity_model("fx-a", 20)])
    context = probe.model_copy(update={"max_tokens": probe.estimated_tokens()})
    engine = make_engine(settings, store, FakeLlmProvider())
    surfaced = engine._surface_unseen(context)
    assert surfaced.focus == "status check"
    assert "New activities since last review" not in surfaced.focus


async def test_submit_feedback_non_pending_raises(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("status check")
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
    draft = await engine.analyze("status check")
    decision = engine.approve(draft.id)
    mutation = decision.report.mutations[0]
    assert isinstance(mutation, CreateWorkout)
    assert mutation.name == "Tempo Session"
    approved = store.get_draft(draft.id)
    assert approved is not None
    assert approved.status is DraftStatus.APPROVED
    assert store.list_decisions() == [decision]


async def test_approve_missing_draft_raises(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(ValueError, match="not found"):
        engine.approve(404)


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
    draft = await engine.analyze("status check")
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


async def test_analyze_reuses_a_provided_context_without_extraction(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    client = FakeIntervalsClient([], [], [], [])
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider, client=client)
    context = CoachContext(focus="how was my week", today=TODAY).model_copy(
        update={"focus": "what do you think?"}
    )
    draft = await engine.analyze("what do you think?", context=context, today=TODAY)
    assert client.calls == []
    assert draft.context.focus == "what do you think?"
    recorded = provider.calls[0]
    assert recorded["json_mode"] is True
    assert [message.role for message in recorded["messages"]] == ["system", "user"]
    assert "what do you think?" in recorded["messages"][1].content


async def test_analyze_includes_history_in_the_prompt(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    history = [
        LlmMessage(role="user", content="past question"),
        LlmMessage(role="assistant", content="past answer"),
    ]
    await engine.analyze("follow up", context=CoachContext(focus="f"), history=history)
    prompt = provider.calls[0]["messages"][1].content
    assert "Recent conversation:" in prompt
    assert "user: past question" in prompt
    assert "assistant: past answer" in prompt


async def test_analyze_keeps_a_stable_prompt_prefix_between_turns(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json()), completion(report_json())])
    engine = make_engine(settings, store, provider)
    context = CoachContext(focus="first", today=TODAY)
    await engine.analyze("first", context=context, today=TODAY)
    moved = context.model_copy(update={"focus": "second"})
    await engine.analyze("second", context=moved, today=TODAY)
    first_prompt = provider.calls[0]["messages"][1].content
    second_prompt = provider.calls[1]["messages"][1].content
    assert first_prompt.split("Current message:")[0] == second_prompt.split("Current message:")[0]
    assert '"focus"' not in first_prompt
    assert "Current message:\nsecond" in second_prompt


async def test_analyze_empty_content_raises_without_writes(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(""), completion(""), completion("")])
    engine = make_engine(settings, store, provider)
    with pytest.raises(LlmError, match="empty content"):
        await engine.analyze("hi", context=CoachContext(focus="f"))
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


def test_recent_history_applies_max_age_cutoff(settings: Settings, tmp_path: Path) -> None:
    clock = FakeClock(datetime(2024, 2, 1, 12, 0, 0, tzinfo=UTC))
    store = CoachStore(tmp_path / "coach.db", clock=clock)
    engine = make_engine(settings, store, FakeLlmProvider())
    draft_id = store.save_draft(
        focus="f", report=DecisionReport(summary="ok"), context=CoachContext(focus="f")
    )
    store.add_feedback(draft_id, "old")
    clock.now = clock.now + timedelta(days=10)
    store.add_feedback(draft_id, "new")
    rows = engine.recent_history(10, max_age_days=5)
    assert [row.feedback.content for row in rows] == ["new"]


async def test_build_context_surfaces_unseen_without_marking(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    context = await engine.build_context("how was my week?", today=TODAY)
    assert "New activities since last review" in context.focus
    assert store.is_activity_seen("fx-a") is False
    assert store.list_drafts() == []


def test_engine_llm_selection_and_switch(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    fake = FakeLlmProvider()
    deepseek = FakeLlmProvider()
    llm = LlmClient(
        settings.model_copy(update={"llm_provider": "fake", "llm_model": "fake-model"}),
        {"fake": fake, "deepseek": deepseek},
    )
    engine = CoachEngine(settings, store, make_intervals_client(), llm)
    assert engine.llm_selection() == ("fake", "fake-model")
    assert engine.select_llm(provider="deepseek") == ("deepseek", "deepseek-flash")
    assert engine.llm_selection() == ("deepseek", "deepseek-flash")
    assert engine.select_llm(model="custom-model") == ("deepseek", "custom-model")
    store.close()


PAST_MUTATION = {
    "action": "create",
    "name": "Copied Example",
    "start_date_local": "2024-01-05",
    "moving_time": 3600,
}


FUTURE_MUTATION = {
    "action": "create",
    "name": "Planned Session",
    "start_date_local": "2024-02-05",
    "moving_time": 3600,
}


async def test_past_dated_mutation_is_retried(settings: Settings, tmp_path: Path) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[PAST_MUTATION])),
            completion(report_json(mutations=[FUTURE_MUTATION])),
        ]
    )
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), provider)
    draft = await engine.analyze("plan my week", today=TODAY)
    assert len(provider.calls) == 2
    created = [
        mutation for mutation in draft.report.mutations if isinstance(mutation, CreateWorkout)
    ]
    assert [mutation.name for mutation in created] == ["Planned Session"]


async def test_past_dated_mutation_exhausts_retries(settings: Settings, tmp_path: Path) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[PAST_MUTATION])),
            completion(report_json(mutations=[PAST_MUTATION])),
            completion(report_json(mutations=[PAST_MUTATION])),
        ]
    )
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), provider)
    with pytest.raises(LlmError, match="validation failed"):
        await engine.analyze("plan my week", today=TODAY)


async def test_approve_rejects_a_mutation_that_is_now_in_the_past(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    race_today = datetime.now(ZoneInfo(settings.app_timezone)).date()
    yesterday = race_today - timedelta(days=1)
    report = DecisionReport.model_validate(
        json.loads(
            report_json(mutations=[{**CREATE_MUTATION, "start_date_local": yesterday.isoformat()}])
        )
    )
    draft_id = store.save_draft(focus="f", report=report, context=CoachContext(focus="f"))
    with pytest.raises(ValueError, match="between today and"):
        engine.approve(draft_id)


async def test_validate_report_rejects_a_past_dated_update(settings: Settings) -> None:
    payload = json.loads(
        report_json(
            mutations=[{"action": "update", "event_id": 7, "start_date_local": "2024-01-01"}]
        )
    )
    with pytest.raises(ValueError, match="between today and"):
        _validate_report(payload, today=date(2024, 2, 1))


async def test_approve_rejects_a_race_mutation_that_is_now_in_the_past(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    race_today = datetime.now(ZoneInfo(settings.app_timezone)).date()
    report = DecisionReport.model_validate(
        json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create_race",
                        "name": "Old Race",
                        "start_date_local": (race_today - timedelta(days=1)).isoformat(),
                        "category": "RACE_A",
                        "moving_time": 3600,
                        "icu_training_load": 90,
                    }
                ]
            )
        )
    )
    draft_id = store.save_draft(focus="f", report=report, context=CoachContext(focus="f"))
    with pytest.raises(ValueError, match="between today and"):
        engine.approve(draft_id)


async def test_apply_refuses_a_decision_that_became_past_dated(
    settings: Settings, tmp_path: Path
) -> None:
    calendar = FakeCalendarClient()
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider(), writer=CalendarWriter(calendar))
    race_today = datetime.now(ZoneInfo(settings.app_timezone)).date()
    report = DecisionReport.model_validate(
        json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create",
                        "name": "Old Session",
                        "start_date_local": (race_today - timedelta(days=1)).isoformat(),
                        "moving_time": 3600,
                    }
                ]
            )
        )
    )
    draft_id = store.save_draft(focus="f", report=report, context=CoachContext(focus="f"))
    decision = store.approve_draft(draft_id)
    with pytest.raises(StaleDecisionError, match="no mutations left to apply"):
        await engine.apply(decision.id)
    assert calendar.created == []


async def test_discard_stale_decisions_removes_now_past_approvals(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    today = datetime.now(ZoneInfo(settings.app_timezone)).date()
    report = DecisionReport.model_validate(
        json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create",
                        "name": "Old Session",
                        "start_date_local": (today - timedelta(days=1)).isoformat(),
                        "moving_time": 3600,
                    }
                ]
            )
        )
    )
    draft_id = store.save_draft(focus="f", report=report, context=CoachContext(focus="f"))
    store.approve_draft(draft_id)
    assert engine.discard_stale_decisions() == [(1, "dates that have passed")]
    assert store.list_unapplied_decisions() == []


def test_validate_report_rejects_placeholder_race_values() -> None:
    payload = json.loads(
        report_json(
            mutations=[
                {
                    "action": "create_race",
                    "name": "Race",
                    "start_date_local": "2099-01-01",
                    "category": "RACE_A",
                    "moving_time": 0,
                    "icu_training_load": 90,
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="duration/load must be real values"):
        _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_placeholder_event_id() -> None:
    payload = json.loads(
        report_json(mutations=[{"action": "update", "event_id": 0, "moving_time": 3600}])
    )
    with pytest.raises(ValueError, match="event_id 0 is a placeholder"):
        _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_digit_string_placeholder_event_id() -> None:
    for placeholder in ("0", "-1", "  "):
        payload = json.loads(
            report_json(
                mutations=[{"action": "update", "event_id": placeholder, "moving_time": 3600}]
            )
        )
        with pytest.raises(PlaceholderMutationError):
            _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_a_race_create_without_load() -> None:
    payload = json.loads(
        report_json(
            mutations=[
                {
                    "action": "create_race",
                    "name": "Race",
                    "start_date_local": "2099-01-01",
                    "category": "RACE_A",
                }
            ]
        )
    )
    with pytest.raises(PlaceholderMutationError, match="duration/load must be real values"):
        _validate_report(payload, today=date(2024, 2, 1))


async def test_apply_skips_past_dated_mutations_and_writes_the_rest(
    settings: Settings, tmp_path: Path
) -> None:
    calendar = FakeCalendarClient()
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider(), writer=CalendarWriter(calendar))
    today = datetime.now(ZoneInfo(settings.app_timezone)).date()
    report = DecisionReport.model_validate(
        json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create",
                        "name": "Past Session",
                        "start_date_local": (today - timedelta(days=1)).isoformat(),
                        "moving_time": 3600,
                    },
                    {
                        "action": "create",
                        "name": "Future Session",
                        "start_date_local": (today + timedelta(days=1)).isoformat(),
                        "moving_time": 3600,
                    },
                ]
            )
        )
    )
    draft_id = store.save_draft(focus="f", report=report, context=CoachContext(focus="f"))
    decision = store.approve_draft(draft_id)
    applied = await engine.apply(decision.id)
    assert applied.decisions[0].skipped == ["past-dated"]
    assert [outcome.target for outcome in applied.decisions[0].outcomes] == ["created"]
    assert [event["name"] for event in calendar.created] == ["Future Session"]


async def test_discard_keeps_a_mixed_decision_with_future_mutations(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    today = datetime.now(ZoneInfo(settings.app_timezone)).date()
    report = DecisionReport.model_validate(
        json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create",
                        "name": "Past Session",
                        "start_date_local": (today - timedelta(days=1)).isoformat(),
                        "moving_time": 3600,
                    },
                    {
                        "action": "create",
                        "name": "Future Session",
                        "start_date_local": (today + timedelta(days=1)).isoformat(),
                        "moving_time": 3600,
                    },
                ]
            )
        )
    )
    draft_id = store.save_draft(focus="f", report=report, context=CoachContext(focus="f"))
    store.approve_draft(draft_id)
    assert engine.discard_stale_decisions() == []
    assert len(store.list_unapplied_decisions()) == 1


async def test_apply_placeholder_only_decision_raises_placeholder_error(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(
        settings, store, FakeLlmProvider(), writer=CalendarWriter(FakeCalendarClient())
    )
    report = DecisionReport.model_validate(
        json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create_race",
                        "name": "Race",
                        "start_date_local": _near_future(),
                        "category": "RACE_A",
                    }
                ]
            )
        )
    )
    draft_id = store.save_draft(focus="f", report=report, context=CoachContext(focus="f"))
    decision = store.approve_draft(draft_id)
    with pytest.raises(PlaceholderMutationError, match="only contains placeholder mutations"):
        await engine.apply(decision.id)


async def test_discard_reports_the_placeholder_reason(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    report = DecisionReport.model_validate(
        json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create_race",
                        "name": "Race",
                        "start_date_local": _near_future(),
                        "category": "RACE_A",
                    }
                ]
            )
        )
    )
    draft_id = store.save_draft(focus="f", report=report, context=CoachContext(focus="f"))
    store.approve_draft(draft_id)
    assert engine.discard_stale_decisions() == [(1, "placeholder values")]


def test_validate_report_rejects_negative_race_values() -> None:
    payload = json.loads(
        report_json(
            mutations=[
                {
                    "action": "create_race",
                    "name": "Race",
                    "start_date_local": "2099-01-01",
                    "category": "RACE_A",
                    "moving_time": -60,
                    "icu_training_load": 90,
                }
            ]
        )
    )
    with pytest.raises(PlaceholderMutationError, match="duration/load must be real values"):
        _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_a_zero_race_distance() -> None:
    payload = json.loads(
        report_json(
            mutations=[
                {
                    "action": "create_race",
                    "name": "Race",
                    "start_date_local": "2099-01-01",
                    "category": "RACE_A",
                    "moving_time": 3600,
                    "distance": 0,
                    "icu_training_load": 90,
                }
            ]
        )
    )
    with pytest.raises(PlaceholderMutationError, match="duration/load must be real values"):
        _validate_report(payload, today=date(2024, 2, 1))


async def test_submit_feedback_includes_the_conversation_history(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json("ok")), completion(report_json("revised"))])
    engine = make_engine(settings, store, provider)
    draft = await engine.analyze("plan my week")
    history = [
        LlmMessage(role="user", content="I can train 4 days and prefer mornings"),
        LlmMessage(role="assistant", content="noted"),
    ]
    await engine.submit_feedback(draft.id, "make it easier", history=history)
    prompt = provider.calls[1]["messages"][1].content
    assert "Recent conversation:" in prompt
    assert "I can train 4 days and prefer mornings" in prompt
    assert "make it easier" in prompt


async def test_history_is_trimmed_to_the_context_budget(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json("ok"))])
    engine = make_engine(settings, store, provider)
    history = [
        LlmMessage(role="user", content="x" * 40000),
        LlmMessage(role="assistant", content="short answer"),
        LlmMessage(role="user", content="recent question"),
    ]
    context = CoachContext(focus="status", max_tokens=200)
    await engine.analyze("status", context=context, history=history)
    prompt = provider.calls[0]["messages"][1].content
    assert "x" * 40000 not in prompt
    assert "recent question" in prompt
    assert "short answer" not in prompt


async def test_history_trimming_keeps_the_newest_turns(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json("ok"))])
    engine = make_engine(settings, store, provider)
    history = [
        LlmMessage(role="user", content="oldest " + "x" * 30000),
        LlmMessage(role="assistant", content="old answer " + "y" * 30000),
        LlmMessage(role="user", content="newest question"),
    ]
    await engine.analyze(
        "status", context=CoachContext(focus="status", max_tokens=200), history=history
    )
    prompt = provider.calls[0]["messages"][1].content
    assert "newest question" in prompt
    assert "oldest " not in prompt
    assert "old answer " not in prompt


async def test_full_history_drop_is_logged(
    settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json("ok"))])
    engine = make_engine(settings, store, provider)
    history = [LlmMessage(role="user", content="x" * 40000)]
    with caplog.at_level("WARNING"):
        await engine.analyze(
            "status", context=CoachContext(focus="status", max_tokens=50), history=history
        )
    assert "dropping the whole conversation history" in caplog.text


def test_validate_report_rejects_placeholder_workout_duration() -> None:
    for moving_time in (0, None):
        payload = json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create",
                        "name": "Workout",
                        "start_date_local": "2024-03-01",
                        "moving_time": moving_time,
                    }
                ]
            )
        )
        with pytest.raises(PlaceholderMutationError, match="duration/load must be real values"):
            _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_zero_workout_load_on_update() -> None:
    payload = json.loads(
        report_json(mutations=[{"action": "update", "event_id": 10001, "icu_training_load": 0}])
    )
    with pytest.raises(PlaceholderMutationError, match="duration/load must be real values"):
        _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_dates_beyond_the_planning_horizon() -> None:
    today = date(2026, 9, 16)
    far = today + timedelta(days=MAX_PLANNING_DAYS + 1)
    near = today + timedelta(days=MAX_PLANNING_DAYS - 1)
    for start, should_raise in ((far, True), (near, False)):
        payload = json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create",
                        "name": "Workout",
                        "start_date_local": start.isoformat(),
                        "moving_time": 3600,
                    }
                ]
            )
        )
        if should_raise:
            with pytest.raises(StaleDecisionError, match="between today and"):
                _validate_report(payload, today=today)
        else:
            _validate_report(payload, today=today)


async def test_total_prompt_stays_under_the_input_ceiling(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json("ok"))])
    engine = make_engine(settings, store, provider)
    context = CoachContext(
        focus="status",
        max_tokens=8192,
        recent_activities=[
            Activity.model_validate(make_activity(f"fx-{index}", index)) for index in range(1, 31)
        ],
    )
    history = [LlmMessage(role="user", content="h" * 6000) for _ in range(3)]
    await engine.analyze("status", context=context, history=history)
    messages = provider.calls[0]["messages"]
    total = sum(estimate_text_tokens(message.content) for message in messages)
    assert total <= INPUT_TOKEN_CEILING
    assert total > estimate_text_tokens(system_prompt(settings))


class _FlakyCalendar(FakeCalendarClient):
    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self._fail_at = fail_at
        self.create_calls = 0

    async def create_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.create_calls += 1
        if self.create_calls == self._fail_at:
            raise IntervalsApiError(503, "calendar down")
        return await super().create_event(payload)


async def test_partial_apply_retry_is_idempotent(settings: Settings, tmp_path: Path) -> None:
    calendar = _FlakyCalendar(fail_at=2)
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider(), writer=CalendarWriter(calendar))
    report = DecisionReport.model_validate(
        json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create",
                        "name": "First Session",
                        "start_date_local": _near_future(1),
                        "moving_time": 3600,
                    },
                    {
                        "action": "create",
                        "name": "Second Session",
                        "start_date_local": _near_future(2),
                        "moving_time": 3600,
                    },
                ]
            )
        )
    )
    draft_id = store.save_draft(focus="f", report=report, context=CoachContext(focus="f"))
    decision = store.approve_draft(draft_id)

    with pytest.raises(IntervalsApiError):
        await engine.apply(decision.id)
    assert [event["name"] for event in calendar.created] == ["First Session"]
    assert [row.id for row in store.list_unapplied_decisions()] == [decision.id]

    await engine.apply(decision.id)
    assert sorted(event["name"] for event in calendar.created) == [
        "First Session",
        "Second Session",
    ]
    assert len(calendar.created) == 2
    assert store.list_unapplied_decisions() == []

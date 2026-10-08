import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.clients.llm import LlmClient, LlmError, LlmMessage
from open_endurance_coach.config import Settings, effective_input_budget
from open_endurance_coach.engine.coach import (
    MAX_PLANNING_DAYS,
    PROMPT_OVERHEAD_TOKENS,
    CoachEngine,
    PlaceholderMutationError,
    StaleProposalError,
    StateDriftError,
    _bad_race_number,
    _drop_reason,
    _validate_report,
)
from open_endurance_coach.errors import InternalError
from open_endurance_coach.extractors.standard import DEFAULT_MAX_TOKENS
from open_endurance_coach.prompts.prompts import (
    CHAT_ONLY_FALLBACK,
    build_messages,
    estimate_user_message_tokens,
    system_prompt,
)
from open_endurance_coach.schemas.context import CoachContext, TrainingWeek
from open_endurance_coach.schemas.decisions import (
    CreateWorkout,
    DecisionReport,
    DeleteWorkout,
    UpdateWorkout,
)
from open_endurance_coach.schemas.intervals import Activity, ActivitySplit
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import MessageRole, ProposalStatus
from open_endurance_coach.tokens import CHARS_PER_TOKEN, estimate_text_tokens
from open_endurance_coach.writer.calendar import CalendarWriter

from .fakes import (
    CREATE_MUTATION,
    TODAY,
    FakeCalendarClient,
    FakeClock,
    FakeIntervalsClient,
    FakeLlmProvider,
    RecordingSleep,
    completion,
    make_activity,
    make_activity_list,
    make_event,
    make_intervals_client,
    near_future,
    report_json,
)

CLOCK = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


def user_messages(store: CoachStore) -> list[str]:
    return [item.content for item in store.list_messages() if item.role is MessageRole.USER]


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
    return CoachEngine(
        settings,
        store,
        client or make_intervals_client(),
        llm,
        writer=writer,
        clock=lambda: CLOCK,
    )


def make_activity_model(activity_id: str, day: int, **overrides: Any) -> Activity:
    payload = make_activity(activity_id, day)
    payload.update(overrides)
    return Activity.model_validate(payload)


async def test_analyze_standard_produces_pending_proposal(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("Analyze this week", today=TODAY)
    assert proposal.status is ProposalStatus.PENDING
    assert proposal.report.summary == "Load stable."
    assert proposal.user_feedback is None
    assert "New activities since last review" in proposal.focus
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
    second_proposal = await second_engine.analyze("Analyze this week", today=TODAY)
    assert "New activities since last review: Evening Ride (2024-01-21)" in second_proposal.focus
    listing = second_proposal.focus.split("New activities since last review")[1]
    assert "Synthetic Workout" not in listing
    assert store.is_activity_seen("fx-z") is True


async def test_resolve_event_dates_backfills_out_of_window_events(
    settings: Settings, tmp_path: Path
) -> None:
    client = make_intervals_client(
        events=[make_event(136743073, "2026-09-17", name="Trail Hill Sharpening")]
    )
    engine = make_engine(
        settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider(), client=client
    )
    context = CoachContext(focus="focus")
    dates = await engine.resolve_event_dates(
        context, [DeleteWorkout(action="delete", event_id=136743073)]
    )
    assert dates == {"136743073": date(2026, 9, 17)}


async def test_resolve_event_dates_uses_recent_events(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    context = CoachContext(
        focus="review yesterday",
        today=TODAY,
        recent_events=[make_event(136743074, "2024-01-31", name="Hill Sharpening")],
    )
    dates = await engine.resolve_event_dates(
        context, [UpdateWorkout(action="update", event_id=136743074, moving_time=3600)]
    )
    assert dates == {"136743074": date(2024, 1, 31)}


async def test_resolve_event_dates_keeps_explicit_mutation_dates(
    settings: Settings, tmp_path: Path
) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    dates = await engine.resolve_event_dates(
        CoachContext(focus="focus"),
        [UpdateWorkout(action="update", event_id=1, start_date_local=date(2026, 9, 21))],
    )
    assert dates == {}


async def test_resolve_event_dates_ignores_unknown_events(
    settings: Settings, tmp_path: Path
) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    dates = await engine.resolve_event_dates(
        CoachContext(focus="focus"), [DeleteWorkout(action="delete", event_id=999)]
    )
    assert dates == {}


async def test_resolve_event_dates_logs_expected_lookup_failure(
    settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with caplog.at_level("WARNING"):
        dates = await engine.resolve_event_dates(
            CoachContext(focus="focus"), [DeleteWorkout(action="delete", event_id=999)]
        )
    assert dates == {}
    assert "could not resolve the date of event 999" in caplog.text


async def test_resolve_event_dates_logs_unexpected_lookup_failure(
    settings: Settings,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_intervals_client()

    async def boom(event_id: str) -> dict[str, Any]:
        raise KeyError(event_id)

    monkeypatch.setattr(client, "get_event", boom)
    engine = make_engine(
        settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider(), client=client
    )
    with caplog.at_level("WARNING"):
        dates = await engine.resolve_event_dates(
            CoachContext(focus="focus"), [DeleteWorkout(action="delete", event_id=999)]
        )
    assert dates == {}
    assert "unexpected error resolving the date of event 999" in caplog.text


async def test_analyze_marks_seen_only_after_success(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion("")] * 6)
    engine = make_engine(settings, store, provider)
    with pytest.raises(LlmError, match="failed after 2 attempts"):
        await engine.analyze("status check", today=TODAY)
    assert store.list_proposals() == []
    assert store.unseen_activity_ids(["fx-a", "fx-b"]) == {"fx-a", "fx-b"}


async def test_analyze_retries_on_schema_invalid_response(
    settings: Settings, tmp_path: Path
) -> None:
    provider = FakeLlmProvider([completion('{"hallucinated": true}'), completion(report_json())])
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), provider)
    proposal = await engine.analyze("status check", today=TODAY)
    assert proposal.report.summary == "Load stable."
    assert len(provider.calls) == 2


async def test_analyze_schema_invalid_exhausts_attempts(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion('{"hallucinated": true}')] * 6)
    engine = make_engine(settings, store, provider)
    with pytest.raises(LlmError, match="failed after 2 attempts"):
        await engine.analyze("status check", today=TODAY)
    assert store.list_proposals() == []


async def test_analyze_falls_back_to_chat_when_the_mutation_is_invalid(
    settings: Settings, tmp_path: Path
) -> None:
    bad = completion(
        report_json(
            mutations=[
                {
                    "action": "create",
                    "name": "Race note",
                    "start_date_local": (TODAY + timedelta(days=1)).isoformat(),
                    "category": "NOTE",
                }
            ]
        )
    )
    chat = completion(report_json("Use the race description for the note."))
    provider = FakeLlmProvider([bad, bad, bad, bad, chat])
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, provider)

    proposal = await engine.analyze("write a note on my race", today=TODAY)

    assert proposal.report.mutations == []
    assert proposal.report.summary == "Use the race description for the note."
    assert len(provider.calls) == 5
    assert "empty mutations list" in provider.calls[-1]["messages"][-1].content


async def test_analyze_uses_deep_extractor_for_deep_focus(
    settings: Settings, tmp_path: Path
) -> None:
    client = make_intervals_client()
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), provider, client=client)
    proposal = await engine.analyze(
        "how did my heart rate improve on hills in the last 3 months", today=TODAY
    )
    assert ("detail", "fx-a") in client.calls
    assert ("activities", "2023-11-03", "2024-02-02") in client.calls
    assert "New activities since last review" in proposal.focus
    assert proposal.context.activity_detail is not None
    assert proposal.context.activity_detail.id == "fx-a"


async def test_analyze_injects_user_feedback(settings: Settings, tmp_path: Path) -> None:
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), provider)
    proposal = await engine.analyze("status check", user_feedback="Legs heavy", today=TODAY)
    assert proposal.user_feedback == "Legs heavy"
    assert "Legs heavy" in provider.calls[0]["messages"][1].content


async def test_review_missing_proposal_raises(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(ValueError, match="not found"):
        engine.review(404)


async def test_submit_feedback_updates_proposal_and_injects_feedback(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider(
        [completion(report_json()), completion(report_json("Revised after feedback."))]
    )
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check", today=TODAY)
    outcome = await engine.submit_feedback(proposal.id, "Legs heavy, RPE 8")
    assert outcome.proposal.id == proposal.id
    assert outcome.report.summary == "Revised after feedback."
    assert outcome.proposal.report.summary == "Revised after feedback."
    assert outcome.proposal.user_feedback == "Legs heavy, RPE 8"
    assert outcome.proposal.status is ProposalStatus.PENDING
    assert "Legs heavy, RPE 8" in provider.calls[1]["messages"][1].content
    assert user_messages(store) == [
        "status check",
        "Legs heavy, RPE 8",
    ]


async def test_submit_feedback_answer_does_not_replace_the_pending_plan(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider(
        [completion(report_json()), completion(report_json("Answer.", intent="analysis"))]
    )
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check", today=TODAY)
    outcome = await engine.submit_feedback(proposal.id, "what about the hike?")
    assert outcome.report.summary == "Answer."
    assert outcome.proposal.report.summary == "Load stable."
    assert user_messages(store) == [
        "status check",
        "what about the hike?",
    ]
    stored = store.get_proposal(proposal.id)
    assert stored is not None
    assert stored.report.summary == "Load stable."


async def test_recent_feedback_keeps_the_answer_report_for_transient_turns(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider(
        [completion(report_json()), completion(report_json("Answer.", intent="analysis"))]
    )
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check", today=TODAY)
    await engine.submit_feedback(proposal.id, "what about the hike?")
    recent = store.recent_messages(1)
    assert recent[0].role is MessageRole.ASSISTANT
    assert recent[0].report is not None
    assert recent[0].report.summary == "Answer."


async def test_refocus_context_keeps_activity_splits(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    splits = [ActivitySplit(label="km 1", distance_m=1000.0, time_s=300)]
    base = CoachContext(focus="hi", activity_splits=splits)
    context = engine.refocus_context(base, "review yesterday", today=TODAY)

    assert [split.label for split in context.activity_splits] == ["km 1"]


async def test_refocus_context_keeps_data_within_budget(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    base = CoachContext(
        focus="hi",
        max_tokens=400,
        training_rollup=[
            TrainingWeek(
                week_start=date(2026, 8, 3) + timedelta(days=7 * index),
                sessions=3,
                time_s=1000,
            )
            for index in range(5)
        ],
    )
    long_message = "x" * ((base.max_tokens + 100) * CHARS_PER_TOKEN)
    context = engine.refocus_context(base, long_message, today=date(2026, 9, 17))
    assert context.data_tokens() <= base.max_tokens
    assert len(context.training_rollup) == len(base.training_rollup)
    assert context.focus == long_message


def test_check_focus_rejects_messages_over_the_limit(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    engine.check_focus("x" * 100)
    over_limit = "x" * ((engine.focus_limit() + 10) * CHARS_PER_TOKEN)
    with pytest.raises(ValueError, match="message too long"):
        engine.check_focus(over_limit)


def test_focus_limit_leaves_room_for_the_context(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    system_tokens = estimate_text_tokens(system_prompt(settings))
    assert system_tokens + engine.focus_limit() + DEFAULT_MAX_TOKENS <= effective_input_budget(
        settings
    )


def test_focus_limit_allows_a_long_question(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    assert engine.focus_limit() >= 20_000


def test_focus_limit_follows_the_selected_model(tmp_path: Path) -> None:
    ovh = Settings(
        intervals_api_key="k", deepseek_api_key="k", llm_input_budget=200_000, _env_file=None
    )
    llm = LlmClient(
        ovh.model_copy(update={"llm_provider": "fake"}),
        {"fake": FakeLlmProvider(), "deepseek": FakeLlmProvider()},
        sleep=RecordingSleep(),
    )
    engine = CoachEngine(ovh, CoachStore(tmp_path / "coach.db"), make_intervals_client(), llm)
    ovh_limit = engine.focus_limit()
    engine.select_llm(provider="deepseek")
    assert engine.focus_limit() > ovh_limit


def test_history_budget_follows_the_selected_model(tmp_path: Path) -> None:
    ovh = Settings(
        intervals_api_key="k", deepseek_api_key="k", llm_input_budget=200_000, _env_file=None
    )
    llm = LlmClient(
        ovh.model_copy(update={"llm_provider": "fake"}),
        {"fake": FakeLlmProvider(), "deepseek": FakeLlmProvider()},
        sleep=RecordingSleep(),
    )
    engine = CoachEngine(ovh, CoachStore(tmp_path / "coach.db"), make_intervals_client(), llm)
    ovh_budget = engine.history_budget()
    engine.select_llm(provider="deepseek")
    assert engine.history_budget() > ovh_budget


def test_history_budget_matches_focus_limit(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    assert engine.history_budget() == engine.focus_limit()


def test_a_budget_too_small_to_chat_is_reported(settings: Settings, tmp_path: Path) -> None:
    tiny = settings.model_copy(update={"llm_input_budget": 5000})
    engine = make_engine(tiny, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(ValueError, match="too small"):
        engine.focus_limit()
    with pytest.raises(ValueError, match="too small"):
        engine.history_budget()


def test_select_llm_rejects_an_unknown_model_without_switching(tmp_path: Path) -> None:
    settings = Settings(intervals_api_key="k", deepseek_api_key="k", _env_file=None)
    llm = LlmClient(
        settings.model_copy(update={"llm_provider": "fake", "llm_model": "fake-model"}),
        {"fake": FakeLlmProvider()},
        sleep=RecordingSleep(),
    )
    engine = CoachEngine(settings, CoachStore(tmp_path / "coach.db"), make_intervals_client(), llm)
    with pytest.raises(ValueError, match="LLM_CONTEXT_WINDOW"):
        engine.select_llm(model="custom-model")
    assert engine.llm_selection() == ("fake", "fake-model")


def test_unknown_model_fails_for_the_budget(tmp_path: Path) -> None:
    settings = Settings(intervals_api_key="k", llm_model="custom-model", _env_file=None)
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(ValueError, match="LLM_CONTEXT_WINDOW"):
        engine.focus_limit()


def test_history_trimming_reserves_the_fallback_message(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    context = CoachContext(focus="status", max_tokens=8192)
    budget = effective_input_budget(settings)
    turn = "h" * 6000
    turns = budget // estimate_text_tokens(turn) + 5
    history = [LlmMessage(role="user", content=turn) for _ in range(turns)]
    system_tokens = estimate_text_tokens(system_prompt(settings))
    trimmed = engine._fit_history(context, history, system_tokens=system_tokens, budget=budget)
    assert trimmed is not None
    assert len(trimmed) < len(history)
    fallback = [
        *build_messages(context, settings, trimmed),
        LlmMessage(role="user", content=CHAT_ONLY_FALLBACK),
    ]
    total = sum(estimate_text_tokens(message.content) for message in fallback)
    assert total <= budget


def test_prompt_overhead_reserve_covers_the_rendered_message(settings: Settings) -> None:
    context = CoachContext(focus="status check", today=date(2026, 9, 17))
    overhead = estimate_user_message_tokens(context) - context.estimated_tokens()
    assert 0 <= overhead < PROMPT_OVERHEAD_TOKENS


def test_request_ceiling_guard_rejects_an_oversized_request(
    settings: Settings, tmp_path: Path
) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    oversized = LlmMessage(
        role="user", content="x" * ((effective_input_budget(settings) + 1) * CHARS_PER_TOKEN)
    )
    with pytest.raises(ValueError, match="request too large"):
        engine._assert_within_ceiling([oversized])


async def test_analyze_rejects_a_message_over_the_focus_limit(
    settings: Settings, tmp_path: Path
) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    over_limit = "x" * ((engine.focus_limit() + 10) * CHARS_PER_TOKEN)
    with pytest.raises(ValueError, match="message too long"):
        await engine.analyze(over_limit, today=TODAY)


async def test_submit_feedback_records_the_message_when_the_llm_fails(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check", today=TODAY)

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise LlmError("provider down")

    monkeypatch.setattr(engine, "_run_llm", boom)
    with pytest.raises(LlmError):
        await engine.submit_feedback(proposal.id, "legs heavy")
    assert user_messages(store) == [
        "status check",
        "legs heavy",
    ]


async def test_submit_feedback_does_not_charge_the_message_against_data_budget(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    base = CoachContext(
        focus="f",
        training_rollup=[
            TrainingWeek(
                week_start=date(2026, 8, 3) + timedelta(days=7 * index),
                sessions=3,
                time_s=1000,
            )
            for index in range(6)
        ],
    )
    proposal_id = store.save_proposal(
        focus="f",
        report=DecisionReport(summary="ok"),
        context=base.model_copy(update={"max_tokens": base.data_tokens() + 100}),
    )
    provider = FakeLlmProvider([completion(report_json("Revised."))])
    engine = make_engine(settings, store, provider)
    long_message = "change the hike block please " * 500
    outcome = await engine.submit_feedback(proposal_id, long_message, focus=long_message)
    assert outcome.proposal.context.user_feedback is None
    assert len(outcome.proposal.context.training_rollup) == len(base.training_rollup)
    assert long_message in provider.calls[0]["messages"][1].content


async def test_refocus_context_keeps_recent_events(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    base = CoachContext(
        focus="f",
        today=TODAY,
        recent_events=[make_event(1, "2024-01-31", name="Prescribed hill session")],
    )
    context = engine.refocus_context(base, "review yesterday", today=TODAY)
    assert [event.name for event in context.recent_events] == ["Prescribed hill session"]


async def test_submit_feedback_persists_feedback_context(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider(
        [completion(report_json()), completion(report_json("Revised after feedback."))]
    )
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check", today=TODAY)
    await engine.submit_feedback(proposal.id, "Legs heavy, RPE 8")
    stored = store.get_proposal(proposal.id)
    assert stored is not None
    assert stored.context.user_feedback == "Legs heavy, RPE 8"


async def test_submit_feedback_trims_an_over_budget_context(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    feedback = "A very long feedback text that must fit the context budget"
    probe = CoachContext(
        focus="status check",
        today=CLOCK.date(),
    )
    context = probe.model_copy(
        update={
            "max_tokens": probe.estimated_tokens() + estimate_text_tokens(feedback) + 5,
            "recent_activities": [],
        }
    )
    proposal_id = store.save_proposal(
        focus="status check", report=DecisionReport(summary="ok"), context=context
    )
    provider = FakeLlmProvider([completion(report_json("Revised."))])
    engine = make_engine(settings, store, provider)
    updated = await engine.submit_feedback(proposal_id, feedback)
    assert updated.report.summary == "Revised."
    assert len(provider.calls) == 1
    prompt = provider.calls[0]["messages"][1].content
    assert feedback in prompt
    assert updated.proposal.context.data_tokens() <= updated.proposal.context.max_tokens


async def test_submit_feedback_falls_back_without_current_proposal_on_budget_overflow(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    big_report = DecisionReport(summary="x" * 400)
    proposal_id = store.save_proposal(
        focus="f", report=big_report, context=CoachContext(focus="f", max_tokens=100)
    )
    provider = FakeLlmProvider([completion(report_json("Revised.", mutations=[CREATE_MUTATION]))])
    engine = make_engine(settings, store, provider)
    updated = await engine.submit_feedback(proposal_id, "make it easier")
    assert user_messages(store) == ["make it easier"]
    assert updated.proposal.context.current_proposal is None
    assert updated.proposal.context.user_feedback == "make it easier"
    assert updated.report.summary == "Revised."


async def test_surface_unseen_falls_back_when_data_exceeds_budget(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    probe = CoachContext(focus="status check", recent_activities=[make_activity_model("fx-a", 20)])
    context = probe.model_copy(update={"max_tokens": 1})
    engine = make_engine(settings, store, FakeLlmProvider())
    surfaced = engine._surface_unseen(context)
    assert surfaced.focus == "status check"
    assert "New activities since last review" not in surfaced.focus


async def test_submit_feedback_non_pending_raises(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check")
    engine.approve(proposal.id)
    with pytest.raises(ValueError, match="pending"):
        await engine.submit_feedback(proposal.id, "too late")


async def test_submit_feedback_missing_proposal_raises(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(ValueError, match="not found"):
        await engine.submit_feedback(404, "feedback")


async def test_approve_records_proposal(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check")
    proposal = engine.approve(proposal.id)
    mutation = proposal.report.mutations[0]
    assert isinstance(mutation, CreateWorkout)
    assert mutation.name == "Tempo Session"
    approved = store.get_proposal(proposal.id)
    assert approved is not None
    assert approved.status is ProposalStatus.UNAPPLIED
    assert store.list_unapplied_proposals() == [proposal]


async def test_approve_keeps_only_the_selected_mutations(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    second = {**CREATE_MUTATION, "name": "Easy Spin"}
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION, second]))])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check")
    proposal = engine.approve(proposal.id, keep=(1,))
    approved_report = proposal.approved_report
    assert approved_report is not None
    assert [m.name for m in approved_report.mutations if isinstance(m, CreateWorkout)] == [
        "Easy Spin"
    ]
    assert store.list_unapplied_proposals() == [proposal]
    stored = store.get_proposal(proposal.id)
    assert stored is not None
    assert stored.status is ProposalStatus.UNAPPLIED
    assert len(stored.report.mutations) == 2


async def test_approve_missing_proposal_raises(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(ValueError, match="not found"):
        engine.approve(404)


async def test_apply_without_writer_raises(settings: Settings, tmp_path: Path) -> None:
    engine = make_engine(settings, CoachStore(tmp_path / "coach.db"), FakeLlmProvider())
    with pytest.raises(InternalError, match="writer"):
        await engine.apply()


async def test_apply_applies_approved_proposals_and_deletes_them(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check")
    engine.approve(proposal.id)
    calendar = FakeCalendarClient()
    writer_engine = make_engine(settings, store, provider, writer=CalendarWriter(calendar))
    report = await writer_engine.apply()
    assert len(report.proposals) == 1
    assert report.proposals[0].proposal_id == 1
    assert report.proposals[0].outcomes[0].target == "created"
    assert len(calendar.created) == 1
    assert store.get_proposal(1) is None
    assert store.list_unapplied_proposals() == []
    assert store.list_messages() != []


async def test_apply_specific_proposal_only(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json()), completion(report_json())])
    engine = make_engine(settings, store, provider)
    first = await engine.analyze("status check", today=TODAY)
    second = await engine.analyze("status check", today=TODAY)
    engine.approve(first.id)
    engine.approve(second.id)
    calendar = FakeCalendarClient()
    writer_engine = make_engine(settings, store, provider, writer=CalendarWriter(calendar))
    report = await writer_engine.apply(proposal_id=second.id)
    assert [item.proposal_id for item in report.proposals] == [second.id]
    assert store.get_proposal(second.id) is None
    kept = store.get_proposal(first.id)
    assert kept is not None
    assert kept.status is ProposalStatus.UNAPPLIED


async def test_apply_a_deleted_proposal_raises(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check", today=TODAY)
    engine.approve(proposal.id)
    writer_engine = make_engine(
        settings, store, provider, writer=CalendarWriter(FakeCalendarClient())
    )
    await writer_engine.apply()
    with pytest.raises(ValueError, match="not found"):
        await writer_engine.apply(proposal_id=1)


async def test_apply_deletes_an_empty_proposal(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check", today=TODAY)
    engine.approve(proposal.id)
    calendar = FakeCalendarClient()
    writer_engine = make_engine(settings, store, provider, writer=CalendarWriter(calendar))
    report = await writer_engine.apply()
    assert report.proposals[0].outcomes == []
    assert calendar.created == []
    assert store.get_proposal(1) is None


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
    proposal = await engine.analyze("what do you think?", context=context, today=TODAY)
    assert client.calls == []
    assert proposal.context.focus == "what do you think?"
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
    provider = FakeLlmProvider([completion("")] * 6)
    engine = make_engine(settings, store, provider)
    with pytest.raises(LlmError, match="empty content"):
        await engine.analyze("hi", context=CoachContext(focus="f"))
    assert store.list_proposals() == []


async def test_recent_history_reads_messages_from_store(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    store.add_message(MessageRole.USER, "legs heavy", report=DecisionReport(summary="ok"))
    rows = engine.recent_history(10)
    assert len(rows) == 1
    assert rows[0].content == "legs heavy"
    assert rows[0].report is not None
    assert rows[0].report.summary == "ok"
    assert engine.recent_history(0) == []


def test_recent_history_applies_max_age_cutoff(settings: Settings, tmp_path: Path) -> None:
    clock = FakeClock(datetime(2024, 2, 1, 12, 0, 0, tzinfo=UTC))
    store = CoachStore(tmp_path / "coach.db", clock=clock)
    engine = make_engine(settings, store, FakeLlmProvider())
    store.add_message(MessageRole.USER, "old")
    clock.now = clock.now + timedelta(days=10)
    store.add_message(MessageRole.USER, "new")
    rows = engine.recent_history(10, max_age_days=5)
    assert [row.content for row in rows] == ["new"]


async def test_build_context_surfaces_unseen_without_marking(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    context = await engine.build_context("how was my week?", today=TODAY)
    assert "New activities since last review" in context.focus
    assert store.is_activity_seen("fx-a") is False
    assert store.list_proposals() == []


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
    with pytest.raises(ValueError, match="LLM_CONTEXT_WINDOW"):
        engine.select_llm(model="custom-model")
    assert engine.llm_selection() == ("deepseek", "deepseek-flash")
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
    proposal = await engine.analyze("plan my week", today=TODAY)
    assert len(provider.calls) == 2
    created = [
        mutation for mutation in proposal.report.mutations if isinstance(mutation, CreateWorkout)
    ]
    assert [mutation.name for mutation in created] == ["Planned Session"]


async def test_past_dated_mutation_exhausts_retries(settings: Settings, tmp_path: Path) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[PAST_MUTATION])),
            completion(report_json(mutations=[PAST_MUTATION])),
            completion(report_json(mutations=[PAST_MUTATION])),
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
    race_today = CLOCK.date()
    yesterday = race_today - timedelta(days=1)
    report = DecisionReport.model_validate(
        json.loads(
            report_json(mutations=[{**CREATE_MUTATION, "start_date_local": yesterday.isoformat()}])
        )
    )
    proposal_id = store.save_proposal(focus="f", report=report, context=CoachContext(focus="f"))
    with pytest.raises(ValueError, match="between today and"):
        engine.approve(proposal_id)


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
    race_today = CLOCK.date()
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
    proposal_id = store.save_proposal(focus="f", report=report, context=CoachContext(focus="f"))
    with pytest.raises(ValueError, match="between today and"):
        engine.approve(proposal_id)


async def test_apply_refuses_a_proposal_that_became_past_dated(
    settings: Settings, tmp_path: Path
) -> None:
    calendar = FakeCalendarClient()
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider(), writer=CalendarWriter(calendar))
    race_today = CLOCK.date()
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
    proposal_id = store.save_proposal(focus="f", report=report, context=CoachContext(focus="f"))
    proposal = store.approve_proposal(proposal_id)
    with pytest.raises(StaleProposalError, match="no mutations left to apply"):
        await engine.apply(proposal.id)
    assert calendar.created == []


async def test_discard_stale_proposals_removes_now_past_approvals(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    today = CLOCK.date()
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
    proposal_id = store.save_proposal(focus="f", report=report, context=CoachContext(focus="f"))
    store.approve_proposal(proposal_id)
    assert engine.discard_stale_proposals() == [(1, "dates that have passed")]
    assert store.list_unapplied_proposals() == []


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
    with pytest.raises(ValueError, match="duration/load/distance must be real values"):
        _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_placeholder_event_id() -> None:
    payload = json.loads(
        report_json(mutations=[{"action": "update", "event_id": 0, "moving_time": 3600}])
    )
    with pytest.raises(ValidationError, match="event_id"):
        _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_digit_string_placeholder_event_id() -> None:
    for placeholder in ("0", "-1", "  "):
        payload = json.loads(
            report_json(
                mutations=[{"action": "update", "event_id": placeholder, "moving_time": 3600}]
            )
        )
        with pytest.raises(ValidationError, match="event_id"):
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
    with pytest.raises(
        PlaceholderMutationError, match="duration/load/distance must be real values"
    ):
        _validate_report(payload, today=date(2024, 2, 1))


async def test_apply_refuses_a_proposal_with_a_past_dated_mutation(
    settings: Settings, tmp_path: Path
) -> None:
    calendar = FakeCalendarClient()
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider(), writer=CalendarWriter(calendar))
    today = CLOCK.date()
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
    proposal_id = store.save_proposal(focus="f", report=report, context=CoachContext(focus="f"))
    proposal = store.approve_proposal(proposal_id)
    with pytest.raises(StateDriftError, match="nothing was written"):
        await engine.apply(proposal.id)
    assert calendar.created == []
    assert [row.id for row in store.list_unapplied_proposals()] == [proposal.id]
    stored = store.get_proposal(proposal.id)
    assert stored is not None


async def test_discard_removes_a_proposal_with_any_dropped_mutation(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    engine = make_engine(settings, store, FakeLlmProvider())
    today = CLOCK.date()
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
    proposal_id = store.save_proposal(focus="f", report=report, context=CoachContext(focus="f"))
    store.approve_proposal(proposal_id)
    assert engine.discard_stale_proposals() == [(1, "dates that have passed")]
    assert store.list_unapplied_proposals() == []


async def test_apply_placeholder_only_proposal_raises_placeholder_error(
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
                        "start_date_local": near_future(),
                        "category": "RACE_A",
                    }
                ]
            )
        )
    )
    proposal_id = store.save_proposal(focus="f", report=report, context=CoachContext(focus="f"))
    proposal = store.approve_proposal(proposal_id)
    with pytest.raises(PlaceholderMutationError, match="only contains placeholder mutations"):
        await engine.apply(proposal.id)


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
                        "start_date_local": near_future(),
                        "category": "RACE_A",
                    }
                ]
            )
        )
    )
    proposal_id = store.save_proposal(focus="f", report=report, context=CoachContext(focus="f"))
    store.approve_proposal(proposal_id)
    assert engine.discard_stale_proposals() == [(1, "placeholder values")]


def test_drop_reason_names_the_single_causes_and_the_mixed_case() -> None:
    assert _drop_reason(["past-dated"]) == "dates that have passed"
    assert _drop_reason(["placeholder"]) == "placeholder values"
    assert _drop_reason(["past-dated", "placeholder"]) == (
        "mutations that no longer match the plan"
    )


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
    with pytest.raises(
        PlaceholderMutationError, match="duration/load/distance must be real values"
    ):
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
    with pytest.raises(
        PlaceholderMutationError, match="duration/load/distance must be real values"
    ):
        _validate_report(payload, today=date(2024, 2, 1))


async def test_submit_feedback_includes_the_conversation_history(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json("ok")), completion(report_json("revised"))])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("plan my week")
    history = [
        LlmMessage(role="user", content="I can train 4 days and prefer mornings"),
        LlmMessage(role="assistant", content="noted"),
    ]
    await engine.submit_feedback(proposal.id, "make it easier", history=history)
    prompt = provider.calls[1]["messages"][1].content
    assert "Recent conversation:" in prompt
    assert "I can train 4 days and prefer mornings" in prompt
    assert "make it easier" in prompt


async def test_history_is_trimmed_to_the_context_budget(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json("ok"))])
    engine = make_engine(settings, store, provider)
    over_budget = effective_input_budget(settings) * CHARS_PER_TOKEN
    history = [
        LlmMessage(role="user", content="zzzzzzzz" + "z" * over_budget),
        LlmMessage(role="assistant", content="short answer"),
        LlmMessage(role="user", content="recent question"),
    ]
    context = CoachContext(focus="status", max_tokens=200)
    await engine.analyze("status", context=context, history=history)
    prompt = provider.calls[0]["messages"][1].content
    assert "zzzzzzzz" not in prompt
    assert "recent question" in prompt
    assert "short answer" not in prompt


async def test_history_trimming_keeps_the_newest_turns(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json("ok"))])
    engine = make_engine(settings, store, provider)
    over_budget = effective_input_budget(settings) * CHARS_PER_TOKEN
    history = [
        LlmMessage(role="user", content="oldest " + "x" * over_budget),
        LlmMessage(role="assistant", content="old answer " + "y" * over_budget),
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
    history = [
        LlmMessage(role="user", content="old question"),
        LlmMessage(
            role="assistant",
            content="x" * ((effective_input_budget(settings) + 1024) * CHARS_PER_TOKEN),
        ),
    ]
    with caplog.at_level("INFO"):
        await engine.analyze(
            "status", context=CoachContext(focus="status", max_tokens=200), history=history
        )
    assert "trimmed the conversation history" in caplog.text


async def test_user_only_history_trim_is_logged(
    settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    engine = make_engine(
        settings,
        CoachStore(tmp_path / "coach.db"),
        FakeLlmProvider([completion(report_json("ok"))]),
    )
    history = [
        LlmMessage(
            role="user", content="x" * ((effective_input_budget(settings) + 1024) * CHARS_PER_TOKEN)
        ),
        LlmMessage(role="user", content="recent question"),
        LlmMessage(role="assistant", content="recent answer"),
    ]
    with caplog.at_level("INFO"):
        await engine.analyze(
            "status", context=CoachContext(focus="status", max_tokens=200), history=history
        )
    assert "trimmed the oldest messages" in caplog.text


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
        with pytest.raises(
            PlaceholderMutationError, match="duration/load/distance must be real values"
        ):
            _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_zero_workout_load_on_update() -> None:
    payload = json.loads(
        report_json(mutations=[{"action": "update", "event_id": 10001, "icu_training_load": 0}])
    )
    with pytest.raises(
        PlaceholderMutationError, match="duration/load/distance must be real values"
    ):
        _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_zero_workout_distance() -> None:
    payload = json.loads(
        report_json(
            mutations=[
                {
                    "action": "create",
                    "name": "Hike",
                    "start_date_local": "2024-03-01",
                    "moving_time": 3600,
                    "distance": 0,
                }
            ]
        )
    )
    with pytest.raises(
        PlaceholderMutationError, match="duration/load/distance must be real values"
    ):
        _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_rejects_zero_workout_distance_on_update() -> None:
    payload = json.loads(
        report_json(mutations=[{"action": "update", "event_id": 10001, "distance": 0}])
    )
    with pytest.raises(
        PlaceholderMutationError, match="duration/load/distance must be real values"
    ):
        _validate_report(payload, today=date(2024, 2, 1))


def test_non_finite_numbers_are_placeholders() -> None:
    assert _bad_race_number(float("nan"), required=False) is True
    assert _bad_race_number(float("inf"), required=False) is True


def test_validate_report_rejects_non_finite_race_distance() -> None:
    for payload in (
        {
            "action": "create_race",
            "name": "Race",
            "start_date_local": "2024-03-01",
            "category": "RACE_B",
        },
        {"action": "update_race", "event_id": 10001},
    ):
        reported = json.loads(report_json(mutations=[{**payload, "distance": float("nan")}]))
        with pytest.raises(
            PlaceholderMutationError, match="duration/load/distance must be real values"
        ):
            _validate_report(reported, today=date(2024, 2, 1))


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
            with pytest.raises(StaleProposalError, match="between today and"):
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
    assert total <= effective_input_budget(settings)
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
                        "start_date_local": near_future(1),
                        "moving_time": 3600,
                    },
                    {
                        "action": "create",
                        "name": "Second Session",
                        "start_date_local": near_future(2),
                        "moving_time": 3600,
                    },
                ]
            )
        )
    )
    proposal_id = store.save_proposal(focus="f", report=report, context=CoachContext(focus="f"))
    proposal = store.approve_proposal(proposal_id)

    with pytest.raises(IntervalsApiError):
        await engine.apply(proposal.id)
    assert [event["name"] for event in calendar.created] == ["First Session"]
    assert [row.id for row in store.list_unapplied_proposals()] == [proposal.id]

    await engine.apply(proposal.id)
    assert sorted(event["name"] for event in calendar.created) == [
        "First Session",
        "Second Session",
    ]
    assert len(calendar.created) == 2
    assert store.list_unapplied_proposals() == []


def test_validate_report_rejects_an_unsafe_event_id() -> None:
    payload = json.loads(
        report_json(
            mutations=[{"action": "update", "event_id": "0/../../athlete/0", "moving_time": 3600}]
        )
    )
    with pytest.raises(ValidationError, match="event_id"):
        _validate_report(payload, today=date(2024, 2, 1))


def test_validate_report_accepts_string_event_ids() -> None:
    payload = json.loads(
        report_json(mutations=[{"action": "update", "event_id": "e20001", "moving_time": 3600}])
    )
    report = _validate_report(payload, today=date(2024, 2, 1))
    mutation = report.mutations[0]
    assert isinstance(mutation, UpdateWorkout)
    assert mutation.event_id == "e20001"


async def test_apply_keeps_the_proposal_when_the_read_back_differs(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check")
    engine.approve(proposal.id)
    calendar = FakeCalendarClient()
    writer_engine = make_engine(settings, store, provider, writer=CalendarWriter(calendar))

    async def failing_read_back(event_id: str) -> dict[str, Any]:
        raise IntervalsApiError(500, "read-back down")

    monkeypatch.setattr(calendar, "get_event", failing_read_back)
    report = await writer_engine.apply()
    assert report.proposals[0].outcomes[0].drift == [
        "read-back failed; planned values were not verified"
    ]
    kept = store.get_proposal(proposal.id)
    assert kept is not None
    assert kept.status is ProposalStatus.UNAPPLIED


async def test_prune_conversation_keeps_proposals_and_dedup(
    settings: Settings, tmp_path: Path
) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    proposal = await engine.analyze("status check")
    removed = engine.prune_conversation()
    assert removed == 2
    assert store.list_messages() == []
    assert store.get_proposal(proposal.id) is not None


async def test_wipe_local_state_clears_everything(settings: Settings, tmp_path: Path) -> None:
    store = CoachStore(tmp_path / "coach.db")
    provider = FakeLlmProvider([completion(report_json())])
    engine = make_engine(settings, store, provider)
    await engine.analyze("status check")

    counts = engine.wipe_local_state()

    assert counts["messages"] == 2
    assert counts["proposals"] == 1
    assert store.list_messages() == []
    assert store.list_proposals() == []

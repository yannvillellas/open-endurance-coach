import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from open_endurance_coach.cli import main as cli_main
from open_endurance_coach.config import Settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import DraftStatus

from .fakes import FakeCalendarClient, FakeLlmProvider, completion, report_json
from .test_cli import CREATE_MUTATION, FakeRunner, decision_of, make_engine

runner = CliRunner()

TODAY = date(2024, 2, 1)


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path) -> Any:
    def build(
        provider: FakeLlmProvider, calendar: FakeCalendarClient | None = None
    ) -> tuple[CoachEngine, CoachStore]:
        engine, store = make_engine(settings, tmp_path, provider, calendar=calendar)
        monkeypatch.setattr(cli_main, "_with_engine", FakeRunner(engine))
        monkeypatch.setattr(cli_main, "get_settings", lambda: settings)
        return engine, store

    return build


def make_fake_prompt(monkeypatch: pytest.MonkeyPatch, script: list[object]) -> None:
    from types import SimpleNamespace

    from open_endurance_coach.cli import chat as cli_chat

    remaining = list(script)

    def ask(prompt: str, *args: Any, **kwargs: Any) -> str:
        step = remaining.pop(0)
        if isinstance(step, BaseException):
            raise step
        assert isinstance(step, str)
        return step

    monkeypatch.setattr(cli_chat, "Prompt", SimpleNamespace(ask=ask))


def _spy_writes(engine: CoachEngine) -> dict[str, int]:
    calls = {"approve": 0, "reject": 0, "apply_write": 0}
    original_approve = engine.approve
    original_reject = engine.reject
    original_apply = engine.apply

    def approve(draft_id: int, *, mutations: Any = None) -> Any:
        calls["approve"] += 1
        return original_approve(draft_id, mutations=mutations)

    def reject(draft_id: int) -> None:
        calls["reject"] += 1
        original_reject(draft_id)

    async def apply(decision_id: int | None = None, *, dry_run: bool = False) -> Any:
        if not dry_run:
            calls["apply_write"] += 1
        return await original_apply(decision_id, dry_run=dry_run)

    engine.approve = approve  # type: ignore[method-assign]
    engine.reject = reject  # type: ignore[method-assign]
    engine.apply = apply  # type: ignore[method-assign]
    return calls


def test_chat_help_lists_commands(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/help\n")
    assert result.exit_code == 0
    assert "/analyze" in result.output
    assert "/clear" in result.output
    assert "/exit" in result.output


def test_chat_exit_says_bye(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/exit\n")
    assert result.exit_code == 0
    assert "bye" in result.output


def test_chat_quit_alias_exits(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/quit\n")
    assert result.exit_code == 0
    assert "bye" in result.output


def test_chat_eof_exits_cleanly(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["chat"], input="hi\n")
    assert result.exit_code == 0
    assert "bye" in result.output


def test_chat_unknown_command_shows_help_without_engine_calls(patched: Any) -> None:
    _, store = patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/bogus\n")
    assert result.exit_code == 0
    assert "/help" in result.output
    assert store.list_drafts() == []


def test_chat_first_free_text_runs_analysis(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json())])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="how was my week?\n")
    assert result.exit_code == 0
    assert "Coach: Load stable." in result.output
    assert store.list_drafts() != []
    assert store.is_activity_seen("fx-a") is True
    assert provider.calls[0]["json_mode"] is True


def test_chat_analyze_with_focus(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze how was my week\n")
    assert result.exit_code == 0
    assert "Coach: Load stable." in result.output
    drafts = store.list_drafts()
    assert len(drafts) == 1
    assert "how was my week" in drafts[0].focus


def test_chat_analyze_without_focus_uses_default(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n")
    assert result.exit_code == 0
    assert store.list_drafts()[0].focus.startswith(cli_main.DEFAULT_ANALYZE_FOCUS)


def test_chat_proposal_yes_writes_calendar(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider, calendar=calendar)
    calls = _spy_writes(engine)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\nyes\n")
    assert result.exit_code == 0
    assert "Apply this to Intervals.icu" in result.output
    assert calls == {"approve": 1, "reject": 0, "apply_write": 1}
    assert len(calendar.created) == 1
    assert decision_of(store, 1).applied_at is not None


def test_chat_proposal_no_writes_nothing(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider, calendar=calendar)
    calls = _spy_writes(engine)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\nno\n")
    assert result.exit_code == 0
    assert calls == {"approve": 0, "reject": 0, "apply_write": 0}
    assert calendar.created == []
    assert store.list_decisions() == []
    assert store.get_draft(1).status is DraftStatus.PENDING


def test_chat_proposal_modification_reruns_and_reasks(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
        ]
    )
    engine, store = patched(provider, calendar=calendar)
    calls = _spy_writes(engine)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\nmake it easier\nyes\n")
    assert result.exit_code == 0
    assert result.output.count("Apply this to Intervals.icu") == 2
    assert calls == {"approve": 1, "reject": 0, "apply_write": 1}
    assert len(calendar.created) == 1
    draft = store.get_draft(1)
    assert draft is not None
    assert draft.user_feedback == "make it easier"


def test_chat_proposal_modification_to_no_mutations_exits_gate(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("No changes needed.")),
            completion("Prose reply."),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\nmake it easier\nhow is it going?\n"
    )
    assert result.exit_code == 0
    assert "No changes proposed anymore." in result.output
    assert "Coach: Prose reply." in result.output


def test_chat_proposal_fuzzy_yes_never_writes(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
        ]
    )
    engine, store = patched(provider, calendar=calendar)
    calls = _spy_writes(engine)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\nyes please\ncancel\n")
    assert result.exit_code == 0
    assert calls == {"approve": 0, "reject": 0, "apply_write": 0}
    assert calendar.created == []
    assert [row.content for row in store.list_feedback(1)] == ["yes please"]


@pytest.mark.parametrize("answer", ["y", "sure", "yes!", "YES SIR"])
def test_chat_proposal_never_writes_without_literal_yes(patched: Any, answer: str) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
        ]
    )
    engine, store = patched(provider, calendar=calendar)
    calls = _spy_writes(engine)
    result = runner.invoke(cli_main.app, ["chat"], input=f"/analyze\n{answer}\ncancel\n")
    assert result.exit_code == 0
    assert calls == {"approve": 0, "reject": 0, "apply_write": 0}
    assert store.list_decisions() == []
    assert store.get_draft(1).status is DraftStatus.PENDING


def test_chat_yes_outside_proposal_never_writes(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion("Sure.")])
    engine, store = patched(provider)
    calls = _spy_writes(engine)
    result = runner.invoke(cli_main.app, ["chat"], input="how was my week?\nyes\n")
    assert result.exit_code == 0
    assert "Coach: Sure." in result.output
    assert calls == {"approve": 0, "reject": 0, "apply_write": 0}
    assert store.list_decisions() == []


def test_chat_cached_turns_are_prose(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion("Prose reply.")])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="how was my week?\nand today?\n")
    assert result.exit_code == 0
    assert "Coach: Prose reply." in result.output
    assert provider.calls[1]["json_mode"] is False
    assert len(store.list_drafts()) == 1


def test_chat_deep_query_refreshes_analysis(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json())])
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app,
        ["chat"],
        input="how much did my heart rate improve on hills over the last 3 months\n",
    )
    assert result.exit_code == 0
    assert "activity_detail" in provider.calls[0]["messages"][1].content
    assert len(store.list_drafts()) == 1


def test_chat_seeds_history_from_feedback(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion("Chat reply.")])
    _, store = patched(provider)
    draft_id = store.save_draft(
        focus="f",
        report=DecisionReport.model_validate(json.loads(report_json("Reconsidered."))),
        context=CoachContext(focus="f"),
    )
    store.add_feedback(draft_id, "legs heavy")
    result = runner.invoke(cli_main.app, ["chat"], input="how was my week?\nand today?\n")
    assert result.exit_code == 0
    second = provider.calls[1]["messages"]
    assert second[2].content == "legs heavy"
    assert "Reconsidered." in second[3].content
    assert second[4].content == "how was my week?"
    assert "Load stable." in second[5].content
    assert second[6].content == "and today?"


def test_chat_session_memory_appends_turns(patched: Any) -> None:
    provider = FakeLlmProvider(
        [completion(report_json()), completion("Answer one."), completion("Answer two.")]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, ["chat"], input="how was my week?\nfirst question\nsecond question\n"
    )
    assert result.exit_code == 0
    third = provider.calls[2]["messages"]
    assert third[2].content == "how was my week?"
    assert "Load stable." in third[3].content
    assert third[4].content == "first question"
    assert third[5].content == "Answer one."
    assert third[6].content == "second question"


def test_chat_gate_feedback_fallback_appends_session_memory(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
            completion("Chat reply."),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app,
        ["chat"],
        input="/analyze\nmake it easier\nyes\nhow is it going?\n",
    )
    assert result.exit_code == 0
    third = provider.calls[2]["messages"]
    assert third[2].content == cli_main.DEFAULT_ANALYZE_FOCUS
    assert third[4].content == "make it easier"
    assert "Reconsidered." in third[5].content
    assert third[6].content == "how is it going?"


def test_chat_ctrl_c_during_confirmation_returns_to_conversing(
    patched: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    make_fake_prompt(monkeypatch, ["/analyze", KeyboardInterrupt(), "/help", EOFError()])
    result = runner.invoke(cli_main.app, ["chat"])
    assert result.exit_code == 0
    assert "Cancelled. Nothing changed." in result.output
    assert "/analyze" in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_chat_eof_during_confirmation_cancels_and_exits(
    patched: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    make_fake_prompt(monkeypatch, ["/analyze", EOFError()])
    result = runner.invoke(cli_main.app, ["chat"])
    assert result.exit_code == 0
    assert "Cancelled. Nothing changed." in result.output
    assert "bye" in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_chat_ctrl_c_while_conversing_exits(patched: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    patched(FakeLlmProvider())
    make_fake_prompt(monkeypatch, [KeyboardInterrupt()])
    result = runner.invoke(cli_main.app, ["chat"])
    assert result.exit_code == 0
    assert "bye" in result.output
    assert "Cancelled" not in result.output


def test_chat_fresh_skips_seeding(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion("Fresh reply.")])
    _, store = patched(provider)
    draft_id = store.save_draft(
        focus="f",
        report=DecisionReport.model_validate(json.loads(report_json())),
        context=CoachContext(focus="f"),
    )
    store.add_feedback(draft_id, "legs heavy")
    result = runner.invoke(
        cli_main.app, ["chat", "--fresh"], input="how was my week?\nand today?\n"
    )
    assert result.exit_code == 0
    assert "Remembering" not in result.output
    second = provider.calls[1]["messages"]
    assert [message.role for message in second] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert second[2].content == "how was my week?"
    assert second[-1].content == "and today?"


def test_chat_shows_seeded_memory_count(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion("Chat reply.")])
    _, store = patched(provider)
    draft_id = store.save_draft(
        focus="f",
        report=DecisionReport.model_validate(json.loads(report_json())),
        context=CoachContext(focus="f"),
    )
    store.add_feedback(draft_id, "legs heavy")
    store.add_feedback(draft_id, "slept badly")
    result = runner.invoke(cli_main.app, ["chat"], input="how was my week?\nand today?\n")
    assert result.exit_code == 0
    assert "Remembering 2 past exchanges." in result.output


def test_chat_clear_wipes_session_memory(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json()),
            completion("Answer two."),
            completion(report_json()),
        ]
    )
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app, ["chat"], input="how was my week?\nsecond question\n/clear\nthird\n"
    )
    assert result.exit_code == 0
    assert "Memory cleared." in result.output
    assert provider.calls[2]["json_mode"] is True
    assert len(store.list_drafts()) == 2


def test_chat_session_trims_to_cap(
    patched: Any, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json()),
            completion("A" * 4000),
            completion("B" * 4000),
            completion("C" * 4000),
        ]
    )
    patched(provider)
    monkeypatch.setattr(
        cli_main,
        "get_settings",
        lambda: settings.model_copy(update={"chat_history_max_tokens": 100}),
    )
    result = runner.invoke(cli_main.app, ["chat"], input="how was my week?\nfirst\nsecond\nthird\n")
    assert result.exit_code == 0
    history = provider.calls[3]["messages"][2:-1]
    assert [message.role for message in history] == ["user", "assistant"]
    assert history[0].content == "second"
    assert history[1].content == "B" * 396


def test_chat_shows_thinking_indicator(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n")
    assert result.exit_code == 0
    assert "Thinking…" in result.output


def test_chat_prose_turn_shows_thinking_indicator(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json()), completion("Prose reply.")]))
    result = runner.invoke(cli_main.app, ["chat"], input="how was my week?\nand today?\n")
    assert result.exit_code == 0
    assert "Thinking…" in result.output


def test_chat_blank_lines_are_skipped(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="\n   \n/help\n")
    assert result.exit_code == 0
    assert "error" not in result.output
    assert "/analyze" in result.output


def test_chat_proposal_question_line_gets_prose_answer_without_replan(
    patched: Any,
) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(mutations=[CREATE_MUTATION])), completion("Explanation.")]
    )
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\nwhat would this train exactly?\nno\n"
    )
    assert result.exit_code == 0
    assert "Coach: Explanation." in result.output
    assert result.output.count("Apply this to Intervals.icu") == 2
    assert len(provider.calls) == 2
    assert len(store.list_drafts()) == 1
    draft = store.get_draft(1)
    assert draft is not None
    assert draft.user_feedback is None


def test_chat_question_plus_change_request_revises_the_plan(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Revised.", mutations=[CREATE_MUTATION])),
        ]
    )
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app,
        ["chat"],
        input="/analyze\nmake it 45 minutes, why did you pick 60?\nyes\n",
    )
    assert result.exit_code == 0
    assert len(provider.calls) == 2
    assert "Coach: Revised." in result.output
    draft = store.get_draft(1)
    assert draft is not None
    assert draft.user_feedback == "make it 45 minutes, why did you pick 60?"
    assert draft.status is DraftStatus.APPROVED


def test_chat_proposal_question_answer_hints_how_to_revise(patched: Any) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(mutations=[CREATE_MUTATION])), completion("Explanation.")]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\nwhat would this train exactly?\nno\n"
    )
    assert result.exit_code == 0
    assert "describe the change" in result.output


def test_chat_proposal_question_answer_includes_the_proposal(patched: Any) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(mutations=[CREATE_MUTATION])), completion("Explanation.")]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\nwhat would this train exactly?\nno\n"
    )
    assert result.exit_code == 0
    user_message = provider.calls[1]["messages"][1].content
    assert "current_proposal" in user_message
    assert "Tempo Session" in user_message


def test_chat_proposal_modification_reshows_report_without_draft_line(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Revised plan.", mutations=[CREATE_MUTATION])),
        ]
    )
    patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\nmake it 4 series instead\nno\n")
    assert result.exit_code == 0
    assert "Coach: Revised plan." in result.output
    assert "Draft #" not in result.output
    assert "Review it" not in result.output


def test_chat_revision_sees_current_proposal(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Revised plan.", mutations=[CREATE_MUTATION])),
        ]
    )
    patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\nmake it 4 series instead\nno\n")
    assert result.exit_code == 0
    user_message = provider.calls[1]["messages"][1].content
    assert "current_proposal" in user_message
    assert "Tempo Session" in user_message


def test_chat_exit_at_gate_leaves_without_llm(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/exit\n")
    assert result.exit_code == 0
    assert "bye" in result.output
    assert len(provider.calls) == 1
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_chat_help_mentions_cancel(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/help\n")
    assert result.exit_code == 0
    assert "cancel" in result.output


def test_chat_proposal_question_budget_overflow_falls_back_to_context(
    patched: Any,
) -> None:
    from open_endurance_coach.chat.gate import PlanSnapshot
    from open_endurance_coach.chat.history import ChatSession
    from open_endurance_coach.chat.state import ChatState
    from open_endurance_coach.cli import chat as cli_chat

    provider = FakeLlmProvider([completion("Explanation.")])
    engine, store = patched(provider)
    draft_id = store.save_draft(
        focus="tight",
        report=DecisionReport.model_validate(json.loads(report_json(mutations=[CREATE_MUTATION]))),
        context=CoachContext(focus="tight", max_tokens=4096),
    )
    session = ChatSession()
    session.context = CoachContext(focus="tight", today=TODAY, max_tokens=25)
    state = ChatState(
        plan=PlanSnapshot(
            action="approve",
            plan_text="Apply this to Intervals.icu:\nProposed changes:\n  - create Tempo Session",
            draft_id=draft_id,
        )
    )
    import asyncio

    asyncio.run(cli_chat._handle_proposal(engine, state, "what is this?", session))
    user_message = provider.calls[0]["messages"][1].content
    assert "current_proposal" not in user_message
    assert provider.calls[0]["json_mode"] is False


async def test_chat_feedback_fallback_keeps_gate_open(patched: Any) -> None:
    from open_endurance_coach.chat.gate import PlanSnapshot
    from open_endurance_coach.chat.history import ChatSession
    from open_endurance_coach.chat.state import ChatState
    from open_endurance_coach.cli import chat as cli_chat

    provider = FakeLlmProvider([completion(report_json("Revised.", mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider)
    big_report = DecisionReport(summary="x" * 400)
    draft_id = store.save_draft(
        focus="f", report=big_report, context=CoachContext(focus="f", max_tokens=100)
    )
    state = ChatState(plan=PlanSnapshot(action="approve", plan_text="plan", draft_id=draft_id))
    session = ChatSession()
    session.context = CoachContext(focus="f", max_tokens=100)
    result = await cli_chat._handle_proposal(engine, state, "make it easier", session)
    assert isinstance(result, ChatState)
    assert result.plan is not None
    assert [row.content for row in store.list_feedback(draft_id)] == ["make it easier"]
    draft = store.get_draft(draft_id)
    assert draft is not None
    assert draft.context.current_proposal is None


def test_chat_proposal_without_draft_errors_gracefully(patched: Any) -> None:
    from open_endurance_coach.chat.gate import PlanSnapshot
    from open_endurance_coach.chat.history import ChatSession
    from open_endurance_coach.chat.state import ChatState
    from open_endurance_coach.cli import chat as cli_chat

    provider = FakeLlmProvider()
    engine, _ = patched(provider)
    state = ChatState(
        plan=PlanSnapshot(
            action="approve",
            plan_text="Apply this to Intervals.icu:",
            draft_id=None,
        )
    )
    import asyncio

    result = asyncio.run(cli_chat._handle_proposal(engine, state, "make it easier", ChatSession()))
    assert isinstance(result, ChatState)
    assert result.plan is None


def test_chat_apply_failure_after_yes_shows_retry_hint(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider, calendar=calendar)

    async def broken_apply(decision_id: int | None = None, *, dry_run: bool = False) -> Any:
        if not dry_run:
            raise RuntimeError("writer exploded")
        return await engine.apply(decision_id, dry_run=True)

    import asyncio

    async def noop() -> None:
        pass

    engine.apply = broken_apply
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\nyes\n")
    assert result.exit_code == 0
    assert "writer exploded" in result.output
    assert "not applied" in result.output
    assert "coach apply" in result.output
    decision = store.get_decision(1)
    assert decision is not None
    assert decision.applied_at is None
    asyncio.run(noop())


def test_chat_sqlite_error_survives_repl(patched: Any) -> None:
    import sqlite3

    provider = FakeLlmProvider([completion(report_json()), completion(report_json())])
    engine, _ = patched(provider)

    async def broken_converse(
        text: str, *, history: Any = None, context: Any = None, today: Any = None
    ) -> str:
        raise sqlite3.OperationalError("database is locked")

    engine.converse = broken_converse
    result = runner.invoke(cli_main.app, ["chat"], input="how was my week?\nand today?\n/help\n")
    assert result.exit_code == 0
    assert "error:" in result.output
    assert "/analyze" in result.output


def test_chat_pure_question_with_digits_stays_a_question(patched: Any) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(mutations=[CREATE_MUTATION])), completion("Explanation.")]
    )
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\nhow long is the 2000m swim?\nno\n"
    )
    assert result.exit_code == 0
    assert "Coach: Explanation." in result.output
    assert len(provider.calls) == 2
    draft = store.get_draft(1)
    assert draft is not None
    assert draft.user_feedback is None


def test_chat_leading_question_with_change_word_stays_a_question(patched: Any) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(mutations=[CREATE_MUTATION])), completion("Explanation.")]
    )
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\nwhy did you add intervals?\nno\n"
    )
    assert result.exit_code == 0
    assert "Coach: Explanation." in result.output
    assert len(provider.calls) == 2
    assert store.list_feedback(1) == []
    draft = store.get_draft(1)
    assert draft is not None
    assert draft.user_feedback is None


def test_chat_help_during_confirmation_skips_llm(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/help\ncancel\n")
    assert result.exit_code == 0
    assert "/analyze" in result.output
    assert len(provider.calls) == 1
    assert store.list_feedback(1) == []


def test_chat_clear_during_confirmation_clears_without_llm(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/clear\ncancel\n")
    assert result.exit_code == 0
    assert "Memory cleared." in result.output
    assert len(provider.calls) == 1
    assert store.list_feedback(1) == []


def test_chat_question_after_clear_mid_gate_still_answered(patched: Any) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(mutations=[CREATE_MUTATION])), completion("Explanation.")]
    )
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\n/clear\nwhat does this train?\nno\n"
    )
    assert result.exit_code == 0
    assert "Coach: Explanation." in result.output
    assert "no cached context" not in result.output
    assert len(provider.calls) == 2
    assert store.list_feedback(1) == []


def test_chat_analyze_during_confirmation_declined_without_llm(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/analyze\nno\n")
    assert result.exit_code == 0
    assert "unavailable while a proposal is open" in result.output
    assert len(provider.calls) == 1
    assert store.list_feedback(1) == []


def test_chat_unknown_command_during_confirmation_skips_llm(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/bogus\ncancel\n")
    assert result.exit_code == 0
    assert "Unknown command." in result.output
    assert len(provider.calls) == 1
    assert store.list_feedback(1) == []

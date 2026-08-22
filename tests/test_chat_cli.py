import json
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
        input="/analyze\nwait, explain\nyes\nhow is it going?\n",
    )
    assert result.exit_code == 0
    third = provider.calls[2]["messages"]
    assert third[2].content == cli_main.DEFAULT_ANALYZE_FOCUS
    assert third[4].content == "wait, explain"
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
    assert len(history) == 1
    assert history[0].content == "B" * 400


def test_chat_blank_lines_are_skipped(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="\n   \n/help\n")
    assert result.exit_code == 0
    assert "error" not in result.output
    assert "/analyze" in result.output

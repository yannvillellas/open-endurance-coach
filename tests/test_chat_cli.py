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


def test_chat_help_lists_commands(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/help\n")
    assert result.exit_code == 0
    assert "/analyze" in result.output
    assert "/feedback" in result.output
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
    patched(FakeLlmProvider([completion("Hello.")]))
    result = runner.invoke(cli_main.app, ["chat"], input="hi\n")
    assert result.exit_code == 0
    assert "bye" in result.output


def test_chat_unknown_command_shows_help_without_engine_calls(patched: Any) -> None:
    _, store = patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/bogus\n")
    assert result.exit_code == 0
    assert "/help" in result.output
    assert store.list_drafts() == []


def test_chat_analyze_saves_draft_with_focus(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze how was my week\n")
    assert result.exit_code == 0
    assert "Draft #1" in result.output
    drafts = store.list_drafts()
    assert len(drafts) == 1
    assert "how was my week" in drafts[0].focus
    assert "New activities since last review" in drafts[0].focus


def test_chat_analyze_without_focus_uses_default(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n")
    assert result.exit_code == 0
    assert "Draft #1" in result.output
    assert store.list_drafts()[0].focus.startswith(cli_main.DEFAULT_ANALYZE_FOCUS)


def test_chat_review_lists_pending_drafts(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json()), completion(report_json("Week two."))]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/analyze\n/review\n")
    assert result.exit_code == 0
    assert "Draft #2" in result.output
    assert "Draft #1" in result.output


def test_chat_review_shows_solicitations(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/review 1\n")
    assert result.exit_code == 0
    assert "RPE missing" in result.output
    assert "/feedback 1" in result.output


def test_chat_feedback_updates_draft(patched: Any) -> None:
    _, store = patched(
        FakeLlmProvider([completion(report_json()), completion(report_json("Revised."))])
    )
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/feedback 1 Legs heavy\n")
    assert result.exit_code == 0
    assert "Draft #1 updated" in result.output
    assert "Revised." in result.output
    draft = store.get_draft(1)
    assert draft is not None
    assert draft.user_feedback == "Legs heavy"
    assert draft.status is DraftStatus.PENDING


def test_chat_bad_draft_id_does_not_kill_repl(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/review abc\n/help\n")
    assert result.exit_code == 0
    assert "draft id must be a number" in result.output
    assert "/analyze" in result.output


def test_chat_review_extra_args_show_usage(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/review 1 2\n")
    assert result.exit_code == 0
    assert "usage" in result.output


def test_chat_feedback_missing_text_shows_usage(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/feedback 1\n")
    assert result.exit_code == 0
    assert "usage" in result.output


def test_chat_feedback_non_numeric_id_shows_usage(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/feedback abc some text\n")
    assert result.exit_code == 0
    assert "usage" in result.output


def test_chat_missing_draft_error_continues(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/review 404\n/help\n")
    assert result.exit_code == 0
    assert "not found" in result.output
    assert "/analyze" in result.output


def test_chat_approve_requires_literal_yes(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/approve 1\nYES\n")
    assert result.exit_code == 0
    assert "Confirm?" in result.output
    assert "approve these mutations" in result.output
    assert "Decision #1 recorded" in result.output
    draft = store.get_draft(1)
    assert draft is not None
    assert draft.status is DraftStatus.APPROVED
    assert len(store.list_decisions()) == 1


def test_chat_approve_no_declines_without_change(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/approve 1\nno\n")
    assert result.exit_code == 0
    assert "Nothing changed." in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_chat_approve_cancel_writes_nothing(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/approve 1\ncancel\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_chat_approve_non_yes_falls_back_to_feedback_then_restates_plan(patched: Any) -> None:
    _, store = patched(
        FakeLlmProvider(
            [
                completion(report_json(mutations=[CREATE_MUTATION])),
                completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
            ]
        )
    )
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\n/approve 1\nwait, explain\nyes\n"
    )
    assert result.exit_code == 0
    assert "Decision #1 recorded" in result.output
    assert result.output.count("approve these mutations") == 2
    assert [row.content for row in store.list_feedback(1)] == ["wait, explain"]
    draft = store.get_draft(1)
    assert draft.report.summary == "Reconsidered."
    assert draft.status is DraftStatus.APPROVED


def test_chat_approve_fuzzy_yes_is_feedback_then_cancel(patched: Any) -> None:
    _, store = patched(
        FakeLlmProvider(
            [
                completion(report_json(mutations=[CREATE_MUTATION])),
                completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
            ]
        )
    )
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\n/approve 1\nyes, but wait\ncancel\n"
    )
    assert result.exit_code == 0
    assert "Cancelled" in result.output
    assert [row.content for row in store.list_feedback(1)] == ["yes, but wait"]
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_chat_approve_requires_draft_id(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/approve\n")
    assert result.exit_code == 0
    assert "usage: /approve <draft_id>" in result.output


def test_chat_reject_requires_literal_yes(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/reject 1\nyes\n")
    assert result.exit_code == 0
    assert "Confirm?" in result.output
    assert "Draft #1 rejected." in result.output
    assert store.get_draft(1).status is DraftStatus.REJECTED


def test_chat_reject_no_keeps_draft_pending(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/reject 1\nno\n")
    assert result.exit_code == 0
    assert "Nothing changed." in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING


def test_chat_apply_dry_run_is_ungated(patched: Any) -> None:
    calendar = FakeCalendarClient()
    _, store = patched(
        FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]),
        calendar=calendar,
    )
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\n/approve 1\nyes\n/apply\n")
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert result.output.count("Confirm?") == 1
    assert calendar.created == []
    assert decision_of(store, 1).applied_at is None


def test_chat_apply_write_requires_literal_yes(patched: Any) -> None:
    calendar = FakeCalendarClient()
    _, store = patched(
        FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]),
        calendar=calendar,
    )
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\n/approve 1\nyes\n/apply 1 --write\nyes\n"
    )
    assert result.exit_code == 0
    assert "Confirm?" in result.output
    assert "Tempo Session" in result.output
    assert len(calendar.created) == 1
    assert decision_of(store, 1).applied_at is not None


def test_chat_apply_write_declined_writes_nothing(patched: Any) -> None:
    calendar = FakeCalendarClient()
    _, store = patched(
        FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]),
        calendar=calendar,
    )
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\n/approve 1\nyes\n/apply 1 --write\nno\n"
    )
    assert result.exit_code == 0
    assert "Nothing changed." in result.output
    assert calendar.created == []
    assert decision_of(store, 1).applied_at is None


def test_chat_apply_discuss_fallback_converses_and_re_prompts(patched: Any) -> None:
    calendar = FakeCalendarClient()
    _, store = patched(
        FakeLlmProvider(
            [
                completion(report_json(mutations=[CREATE_MUTATION])),
                completion("Let me explain."),
            ]
        ),
        calendar=calendar,
    )
    result = runner.invoke(
        cli_main.app,
        ["chat"],
        input="/analyze\n/approve 1\nyes\n/apply --write\nwhat does update mean?\nno\n",
    )
    assert result.exit_code == 0
    assert "Coach: Let me explain." in result.output
    assert result.output.count("Confirm?") == 3
    assert calendar.created == []
    assert decision_of(store, 1).applied_at is None


def test_chat_apply_bad_args_show_usage(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="/apply abc --write\n")
    assert result.exit_code == 0
    assert "usage: /apply [decision_id] [--write]" in result.output


def test_chat_free_text_converses_without_drafts(patched: Any) -> None:
    provider = FakeLlmProvider([completion("Keep it steady.")])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="how was my week?\n")
    assert result.exit_code == 0
    assert "Coach: Keep it steady." in result.output
    assert store.list_drafts() == []
    assert store.is_activity_seen("fx-a") is False
    assert provider.calls[0]["json_mode"] is False


def test_chat_seeds_history_from_feedback(patched: Any) -> None:
    provider = FakeLlmProvider([completion("Chat reply.")])
    _, store = patched(provider)
    draft_id = store.save_draft(
        focus="f",
        report=DecisionReport.model_validate(json.loads(report_json("Reconsidered."))),
        context=CoachContext(focus="f"),
    )
    store.add_feedback(draft_id, "legs heavy")
    result = runner.invoke(cli_main.app, ["chat"], input="how is it going?\n")
    assert result.exit_code == 0
    messages = provider.calls[-1]["messages"]
    assert [message.role for message in messages] == ["system", "user", "user", "assistant", "user"]
    assert messages[2].content == "legs heavy"
    assert "Reconsidered." in messages[3].content
    assert messages[4].content == "how is it going?"


def test_chat_session_memory_appends_turns(patched: Any) -> None:
    provider = FakeLlmProvider([completion("Answer one."), completion("Answer two.")])
    patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="first question\nsecond question\n")
    assert result.exit_code == 0
    second = provider.calls[1]["messages"]
    assert [message.role for message in second] == ["system", "user", "user", "assistant", "user"]
    assert second[2].content == "first question"
    assert second[3].content == "Answer one."
    assert second[4].content == "second question"


def test_chat_deep_query_refreshes_context(patched: Any) -> None:
    provider = FakeLlmProvider([completion("Hills improving.")])
    patched(provider)
    result = runner.invoke(
        cli_main.app,
        ["chat"],
        input="how much did my heart rate improve on hills over the last 3 months\n",
    )
    assert result.exit_code == 0
    assert "activity_detail" in provider.calls[0]["messages"][1].content


def test_chat_analyze_resets_cached_context(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion("Chat reply.")])
    patched(provider)
    result = runner.invoke(cli_main.app, ["chat"], input="/analyze\nhow is it going?\n")
    assert result.exit_code == 0
    assert "New activities since last review" not in provider.calls[1]["messages"][1].content


def test_chat_feedback_appends_to_session_history(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json()),
            completion(report_json("Reconsidered.")),
            completion("Chat reply."),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\n/feedback 1 legs heavy\nhow is it going?\n"
    )
    assert result.exit_code == 0
    third = provider.calls[2]["messages"]
    assert third[2].content == "legs heavy"
    assert "Reconsidered." in third[3].content
    assert third[4].content == "how is it going?"


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
        input="/analyze\n/approve 1\nwait, explain\nyes\nhow is it going?\n",
    )
    assert result.exit_code == 0
    third = provider.calls[2]["messages"]
    assert third[2].content == "wait, explain"
    assert "Reconsidered." in third[3].content
    assert third[4].content == "how is it going?"


def test_chat_session_trims_to_cap(
    patched: Any, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    provider = FakeLlmProvider(
        [completion("A" * 4000), completion("B" * 4000), completion("C" * 4000)]
    )
    patched(provider)
    monkeypatch.setattr(
        cli_main,
        "get_settings",
        lambda: settings.model_copy(update={"chat_history_max_tokens": 100}),
    )
    result = runner.invoke(cli_main.app, ["chat"], input="first\nsecond\nthird\n")
    assert result.exit_code == 0
    history = provider.calls[2]["messages"][2:-1]
    assert len(history) == 1
    assert history[0].content == "B" * 400


def test_chat_blank_lines_are_skipped(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="\n   \n/help\n")
    assert result.exit_code == 0
    assert "error" not in result.output
    assert "/analyze" in result.output


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


def test_chat_ctrl_c_during_confirmation_returns_to_conversing(
    patched: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    make_fake_prompt(
        monkeypatch, ["/analyze", "/approve 1", KeyboardInterrupt(), "/help", EOFError()]
    )
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
    make_fake_prompt(monkeypatch, ["/analyze", "/approve 1", EOFError()])
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


def test_chat_approve_on_approved_draft_refuses_before_prompting(
    patched: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    make_fake_prompt(monkeypatch, ["/analyze", "/approve 1", "yes", "/approve 1", EOFError()])
    result = runner.invoke(cli_main.app, ["chat"])
    assert result.exit_code == 0
    assert result.output.count("Confirm?") == 1
    assert "only pending drafts can be approved" in result.output
    assert store.get_draft(1).status is DraftStatus.APPROVED
    assert len(store.list_decisions()) == 1


def test_chat_reject_on_approved_draft_refuses_before_prompting(
    patched: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    make_fake_prompt(monkeypatch, ["/analyze", "/approve 1", "yes", "/reject 1", EOFError()])
    result = runner.invoke(cli_main.app, ["chat"])
    assert result.exit_code == 0
    assert result.output.count("Confirm?") == 1
    assert "only pending drafts can be rejected" in result.output
    assert store.get_draft(1).status is DraftStatus.APPROVED


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


def test_chat_yes_outside_confirmation_never_writes(patched: Any) -> None:
    provider = FakeLlmProvider([completion("Sure.")])
    engine, store = patched(provider)
    calls = _spy_writes(engine)
    result = runner.invoke(cli_main.app, ["chat"], input="yes\n")
    assert result.exit_code == 0
    assert "Coach: Sure." in result.output
    assert calls == {"approve": 0, "reject": 0, "apply_write": 0}
    assert store.list_decisions() == []


def test_chat_fuzzy_yes_at_gate_never_writes(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
        ]
    )
    engine, store = patched(provider)
    calls = _spy_writes(engine)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\n/approve 1\nyes please\ncancel\n"
    )
    assert result.exit_code == 0
    assert calls["approve"] == 0
    assert store.list_decisions() == []
    assert [row.content for row in store.list_feedback(1)] == ["yes please"]


@pytest.mark.parametrize("answer", ["y", "sure", "yes!", "YES SIR", ""])
def test_chat_approve_never_writes_without_literal_yes(patched: Any, answer: str) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
        ]
    )
    engine, store = patched(provider)
    calls = _spy_writes(engine)
    result = runner.invoke(
        cli_main.app, ["chat"], input=f"/analyze\n/approve 1\n{answer}\ncancel\n"
    )
    assert result.exit_code == 0
    assert calls["approve"] == 0
    assert store.list_decisions() == []


def test_chat_apply_write_requires_gate_yes(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider, calendar=calendar)
    calls = _spy_writes(engine)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\n/approve 1\nyes\n/apply --write\nno\n"
    )
    assert result.exit_code == 0
    assert calls["apply_write"] == 0
    assert calendar.created == []
    assert decision_of(store, 1).applied_at is None


def test_chat_reject_requires_gate_yes(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion(report_json("R."))])
    engine, store = patched(provider)
    calls = _spy_writes(engine)
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\n/reject 1\nYES, DO IT\ncancel\n"
    )
    assert result.exit_code == 0
    assert calls["reject"] == 0
    assert store.get_draft(1).status is DraftStatus.PENDING


def test_chat_only_literal_yes_reaches_the_gated_calls(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider, calendar=calendar)
    calls = _spy_writes(engine)
    result = runner.invoke(
        cli_main.app,
        ["chat"],
        input="/analyze\n/approve 1\nyes\n/apply 1 --write\nyes\n",
    )
    assert result.exit_code == 0
    assert calls == {"approve": 1, "reject": 0, "apply_write": 1}
    assert len(calendar.created) == 1
    assert decision_of(store, 1).applied_at is not None

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner

from open_endurance_coach.chat.history import ChatSession
from open_endurance_coach.cli import chat as cli_chat
from open_endurance_coach.cli import main as cli_main
from open_endurance_coach.clients.llm import LlmClient
from open_endurance_coach.config import Settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import DraftStatus

from .fakes import (
    CREATE_MUTATION,
    FakeCalendarClient,
    FakeLlmProvider,
    FakeRunner,
    completion,
    decision_of,
    make_engine,
    make_intervals_client,
    report_json,
)

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
    calls = {"approve": 0, "apply_write": 0}
    original_approve = engine.approve
    original_apply = engine.apply

    def approve(draft_id: int) -> Any:
        calls["approve"] += 1
        return original_approve(draft_id)

    async def apply(decision_id: int | None = None) -> Any:
        calls["apply_write"] += 1
        return await original_apply(decision_id)

    engine.approve = approve  # type: ignore[method-assign]
    engine.apply = apply  # type: ignore[method-assign]
    return calls


def test_chat_provider_option_threads_to_engine(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    from open_endurance_coach.cli import chat as cli_chat

    engine, _ = make_engine(settings, tmp_path, FakeLlmProvider())
    captured: dict[str, Any] = {}

    async def fake_with_engine(
        callback: Callable[[CoachEngine], Awaitable[None]],
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        captured["provider"] = provider
        captured["model"] = model
        await callback(engine)

    async def fake_run_chat(
        engine: CoachEngine, settings: Settings
    ) -> None:
        return None

    monkeypatch.setattr(cli_main, "_with_engine", fake_with_engine)
    monkeypatch.setattr(cli_main, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_chat, "run_chat", fake_run_chat)

    result = runner.invoke(cli_main.app, ["--provider", "deepseek"])
    assert result.exit_code == 0
    assert captured == {"provider": "deepseek", "model": None}


def test_chat_startup_shows_provider_and_model(patched: Any, settings: Settings) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="/exit\n")
    assert result.exit_code == 0
    assert f"Using fake ({settings.llm_model})." in result.output


def test_chat_provider_command_shows_current(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="/provider\n/exit\n")
    assert result.exit_code == 0
    assert "Using fake (" in result.output


def test_chat_provider_command_switches_and_next_analysis_uses_it(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    deepseek = FakeLlmProvider([completion(report_json())])
    llm = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": FakeLlmProvider(), "deepseek": deepseek},
    )
    store = CoachStore(tmp_path / "coach.db")
    engine = CoachEngine(settings, store, make_intervals_client(), llm)
    monkeypatch.setattr(cli_main, "_with_engine", FakeRunner(engine))
    monkeypatch.setattr(cli_main, "get_settings", lambda: settings)
    result = runner.invoke(cli_main.app, [], input="/provider deepseek\nanalyze my week\n/exit\n")
    assert result.exit_code == 0
    assert "Using deepseek (deepseek-flash)." in result.output
    assert deepseek.calls[-1]["model"] == "deepseek-flash"
    store.close()


def test_chat_model_command_sets_model(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    fake = FakeLlmProvider([completion(report_json())])
    llm = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": fake},
    )
    store = CoachStore(tmp_path / "coach.db")
    engine = CoachEngine(settings, store, make_intervals_client(), llm)
    monkeypatch.setattr(cli_main, "_with_engine", FakeRunner(engine))
    monkeypatch.setattr(cli_main, "get_settings", lambda: settings)
    result = runner.invoke(cli_main.app, [], input="/model my-model\nanalyze my week\n/exit\n")
    assert result.exit_code == 0
    assert "Using fake (my-model)." in result.output
    assert fake.calls[-1]["model"] == "my-model"
    store.close()


def test_chat_model_command_escapes_markup(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="/model bad[red]\n/exit\n")
    assert result.exit_code == 0
    assert "bad[red]" in result.output


def test_chat_provider_command_unknown_provider_prints_error(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="/provider nope\n/exit\n")
    assert result.exit_code == 0
    assert "Unknown LLM provider" in result.output
    assert "bye" in result.output


def test_chat_provider_during_confirmation_switches_without_llm(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, [], input="analyze my week\n/provider fake\ncancel\n")
    assert result.exit_code == 0
    assert "Using fake (" in result.output
    assert len(provider.calls) == 1
    assert store.list_feedback(1) == []


def test_chat_help_lists_commands(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="/help\n")
    assert result.exit_code == 0
    assert "/provider" in result.output
    assert "/forget" in result.output
    assert "/exit" in result.output


def test_bare_coach_starts_chat(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="/exit\n")
    assert result.exit_code == 0
    assert "Chat with the coach" in result.output
    assert "bye" in result.output


def test_bare_coach_forwards_provider_option(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    from collections.abc import Awaitable, Callable

    engine, _ = make_engine(settings, tmp_path, FakeLlmProvider())
    captured: dict[str, Any] = {}

    async def fake_with_engine(
        callback: Callable[[CoachEngine], Awaitable[None]],
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        captured["provider"] = provider
        captured["model"] = model
        await callback(engine)

    monkeypatch.setattr(cli_main, "_with_engine", fake_with_engine)
    monkeypatch.setattr(cli_main, "get_settings", lambda: settings)
    result = runner.invoke(cli_main.app, ["-p", "deepseek"], input="/exit\n")
    assert result.exit_code == 0
    assert captured == {"provider": "deepseek", "model": None}


def test_chat_exit_says_bye(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="/exit\n")
    assert result.exit_code == 0
    assert "bye" in result.output


def test_chat_quit_alias_exits(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="/quit\n")
    assert result.exit_code == 0
    assert "bye" in result.output


def test_chat_eof_exits_cleanly(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, [], input="hi\n")
    assert result.exit_code == 0
    assert "bye" in result.output


def test_chat_unknown_command_shows_help_without_engine_calls(patched: Any) -> None:
    _, store = patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="/bogus\n")
    assert result.exit_code == 0
    assert "/help" in result.output
    assert store.list_drafts() == []


def test_chat_first_free_text_runs_analysis(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json())])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, [], input="how was my week?\n")
    assert result.exit_code == 0
    assert "Coach: Load stable." in result.output
    assert store.list_drafts() != []
    assert store.is_activity_seen("fx-a") is True
    assert provider.calls[0]["json_mode"] is True


def test_chat_free_text_becomes_the_analysis_focus(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, [], input="analyze the week how was my week\n")
    assert result.exit_code == 0
    assert "Coach: Load stable." in result.output
    drafts = store.list_drafts()
    assert len(drafts) == 1
    assert drafts[0].focus.splitlines()[0] == "analyze the week how was my week"


def test_chat_first_free_text_keeps_the_exact_focus(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, [], input="how was my week?\n")
    assert result.exit_code == 0
    assert store.list_drafts()[0].focus.splitlines()[0] == "how was my week?"


def test_chat_proposal_yes_writes_calendar(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider, calendar=calendar)
    calls = _spy_writes(engine)
    result = runner.invoke(cli_main.app, [], input="analyze my week\nyes\n")
    assert result.exit_code == 0
    assert "Apply this to Intervals.icu" in result.output
    assert calls == {"approve": 1, "apply_write": 1}
    assert len(calendar.created) == 1
    assert decision_of(store, 1).applied_at is not None


def test_chat_proposal_no_writes_nothing(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider, calendar=calendar)
    calls = _spy_writes(engine)
    result = runner.invoke(cli_main.app, [], input="analyze my week\nno\n")
    assert result.exit_code == 0
    assert calls == {"approve": 0, "apply_write": 0}
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
    result = runner.invoke(cli_main.app, [], input="analyze my week\nmake it easier\nyes\n")
    assert result.exit_code == 0
    assert result.output.count("Apply this to Intervals.icu") == 2
    assert calls == {"approve": 1, "apply_write": 1}
    assert len(calendar.created) == 1
    draft = store.get_draft(1)
    assert draft is not None
    assert draft.user_feedback == "make it easier"


def test_chat_proposal_modification_to_no_mutations_exits_gate(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("No changes needed.")),
            completion(report_json("Prose reply.")),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="analyze my week\nmake it easier\nhow is it going?\n"
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
    result = runner.invoke(cli_main.app, [], input="analyze my week\nyes please\ncancel\n")
    assert result.exit_code == 0
    assert calls == {"approve": 0, "apply_write": 0}
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
    result = runner.invoke(cli_main.app, [], input=f"analyze my week\n{answer}\ncancel\n")
    assert result.exit_code == 0
    assert calls == {"approve": 0, "apply_write": 0}
    assert store.list_decisions() == []
    assert store.get_draft(1).status is DraftStatus.PENDING


def test_chat_yes_outside_proposal_never_writes(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion(report_json("Sure."))])
    engine, store = patched(provider)
    calls = _spy_writes(engine)
    result = runner.invoke(cli_main.app, [], input="how was my week?\nyes\n")
    assert result.exit_code == 0
    assert "Coach: Sure." in result.output
    assert calls == {"approve": 0, "apply_write": 0}
    assert store.list_decisions() == []


def test_chat_deep_query_refreshes_analysis(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json())])
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app,
        [],
        input="how much did my heart rate improve on hills over the last 3 months\n",
    )
    assert result.exit_code == 0
    assert '"activity_detail": null' not in provider.calls[0]["messages"][1].content
    assert len(store.list_drafts()) == 1


def test_chat_seeds_history_from_feedback(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion(report_json())])
    _, store = patched(provider)
    draft_id = store.save_draft(
        focus="f",
        report=DecisionReport.model_validate(json.loads(report_json("Reconsidered."))),
        context=CoachContext(focus="f"),
    )
    store.add_feedback(draft_id, "legs heavy")
    result = runner.invoke(cli_main.app, [], input="how was my week?\nand today?\n")
    assert result.exit_code == 0
    prompt = provider.calls[1]["messages"][1].content
    assert "Recent conversation:" in prompt
    assert "legs heavy" in prompt
    assert "Reconsidered." in prompt


def test_chat_seed_passes_max_age_from_settings(
    patched: Any, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion(report_json("Chat reply."))])
    engine, _ = patched(provider)
    monkeypatch.setattr(
        cli_main,
        "get_settings",
        lambda: settings.model_copy(update={"chat_history_max_age_days": 30}),
    )
    seen: list[Any] = []
    original = engine.recent_history

    def spy(limit: int, *, max_age_days: int | None = None) -> Any:
        seen.append((limit, max_age_days))
        return original(limit, max_age_days=max_age_days)

    monkeypatch.setattr(engine, "recent_history", spy)
    result = runner.invoke(cli_main.app, [], input="how was my week?\n")
    assert result.exit_code == 0
    assert seen == [(10, 30)]


def test_chat_session_memory_appends_turns(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()) for _ in range(3)])
    patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="how was my week?\nfirst question\nsecond question\n"
    )
    assert result.exit_code == 0
    prompt = provider.calls[2]["messages"][1].content
    assert "Recent conversation:" in prompt
    assert "user: how was my week?" in prompt
    assert "user: first question" in prompt
    assert "second question" in prompt


def test_chat_gate_feedback_fallback_appends_session_memory(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
            completion(report_json("Going well.")),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app,
        [],
        input="analyze my week\nmake it easier\nyes\nhow is it going?\n",
    )
    assert result.exit_code == 0
    prompt = provider.calls[2]["messages"][1].content
    assert "Recent conversation:" in prompt
    assert "make it easier" in prompt
    assert "Reconsidered." in prompt


def test_chat_ctrl_c_during_confirmation_returns_to_conversing(
    patched: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    make_fake_prompt(monkeypatch, ["analyze my week", KeyboardInterrupt(), "/help", EOFError()])
    result = runner.invoke(cli_main.app, [])
    assert result.exit_code == 0
    assert "Cancelled. Nothing changed." in result.output
    assert "/provider" in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_chat_eof_during_confirmation_cancels_and_exits(
    patched: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    make_fake_prompt(monkeypatch, ["analyze my week", EOFError()])
    result = runner.invoke(cli_main.app, [])
    assert result.exit_code == 0
    assert "Cancelled. Nothing changed." in result.output
    assert "bye" in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_chat_ctrl_c_while_conversing_exits(patched: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    patched(FakeLlmProvider())
    make_fake_prompt(monkeypatch, [KeyboardInterrupt()])
    result = runner.invoke(cli_main.app, [])
    assert result.exit_code == 0
    assert "bye" in result.output
    assert "Cancelled" not in result.output


def test_chat_startup_keeps_history_when_window_is_zero(
    patched: Any, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    provider = FakeLlmProvider([completion(report_json())])
    _, store = patched(provider)
    draft_id = store.save_draft(
        focus="f",
        report=DecisionReport.model_validate(json.loads(report_json())),
        context=CoachContext(focus="f"),
    )
    store.add_feedback(draft_id, "legs heavy")
    monkeypatch.setattr(
        cli_main, "get_settings", lambda: settings.model_copy(update={"history_days": 0})
    )
    result = runner.invoke(cli_main.app, [], input="how was my week?\n")
    assert result.exit_code == 0
    assert "Remembering 1 past exchange" in result.output
    assert "Pruned" not in result.output


def test_chat_shows_seeded_memory_count(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion(report_json())])
    _, store = patched(provider)
    draft_id = store.save_draft(
        focus="f",
        report=DecisionReport.model_validate(json.loads(report_json())),
        context=CoachContext(focus="f"),
    )
    store.add_feedback(draft_id, "legs heavy")
    store.add_feedback(draft_id, "slept badly")
    result = runner.invoke(cli_main.app, [], input="how was my week?\nand today?\n")
    assert result.exit_code == 0
    assert "Remembering 2 past exchanges." in result.output


def test_chat_forget_wipes_stored_history_and_memory(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()) for _ in range(3)])
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="how was my week?\nsecond question\n/forget\nthird\n"
    )
    assert result.exit_code == 0
    assert "Forgot" in result.output
    assert "Recent conversation:" not in provider.calls[2]["messages"][1].content
    drafts = store.list_drafts()
    assert len(drafts) == 1
    assert drafts[0].focus.startswith("third")


def test_chat_session_trims_to_cap(
    patched: Any, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    provider = FakeLlmProvider([completion(report_json()) for _ in range(4)])
    patched(provider)
    monkeypatch.setattr(
        cli_main,
        "get_settings",
        lambda: settings.model_copy(update={"chat_history_max_tokens": 100}),
    )
    result = runner.invoke(
        cli_main.app,
        [],
        input=f"how was my week?\n{'A' * 4000}\n{'B' * 4000}\n{'C' * 4000}\n",
    )
    assert result.exit_code == 0
    prompt = provider.calls[3]["messages"][1].content
    assert "Recent conversation:" in prompt
    assert "A" * 4000 not in prompt


def test_chat_shows_thinking_indicator(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, [], input="analyze my week\n")
    assert result.exit_code == 0
    assert "Thinking…" in result.output


def test_chat_blank_lines_are_skipped(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="\n   \n/help\n")
    assert result.exit_code == 0
    assert "error" not in result.output
    assert "/provider" in result.output


def test_chat_mid_session_planning_request_opens_a_proposal(patched: Any) -> None:
    provider = FakeLlmProvider(
        [completion(report_json()), completion(report_json(mutations=[CREATE_MUTATION]))]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="how was my week?\nplan a rest run tomorrow\ncancel\n"
    )
    assert result.exit_code == 0
    assert len(provider.calls) == 2
    assert "Confirm? Reply with exactly yes or no" in result.output


def test_chat_proposal_question_line_gets_an_answer_without_replan(
    patched: Any,
) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Explanation.")),
        ]
    )
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="analyze my week\nwhat would this train exactly?\nno\n"
    )
    assert result.exit_code == 0
    assert "Coach: Explanation." in result.output
    assert result.output.count("Apply this to Intervals.icu") == 2
    assert len(provider.calls) == 2
    assert len(store.list_drafts()) == 2
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
        [],
        input="analyze my week\nmake it 45 minutes, why did you pick 60?\nyes\n",
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
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Explanation.")),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="analyze my week\nwhat would this train exactly?\nno\n"
    )
    assert result.exit_code == 0
    assert "describe the change" in result.output


def test_chat_proposal_question_answer_includes_the_proposal(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Explanation.")),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="analyze my week\nwhat would this train exactly?\nno\n"
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
    result = runner.invoke(
        cli_main.app, [], input="analyze my week\nmake it 4 series instead\nno\n"
    )
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
    result = runner.invoke(
        cli_main.app, [], input="analyze my week\nmake it 4 series instead\nno\n"
    )
    assert result.exit_code == 0
    user_message = provider.calls[1]["messages"][1].content
    assert "current_proposal" in user_message
    assert "Tempo Session" in user_message


def test_chat_exit_at_gate_leaves_without_llm(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, [], input="analyze my week\n/exit\n")
    assert result.exit_code == 0
    assert "bye" in result.output
    assert len(provider.calls) == 1
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_chat_help_mentions_cancel(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, [], input="/help\n")
    assert result.exit_code == 0
    assert "cancel" in result.output


def test_chat_proposal_question_budget_overflow_falls_back_to_context(
    patched: Any,
) -> None:
    from open_endurance_coach.chat.gate import PlanSnapshot
    from open_endurance_coach.chat.history import ChatSession
    from open_endurance_coach.chat.state import ChatState
    from open_endurance_coach.cli import chat as cli_chat

    provider = FakeLlmProvider([completion(report_json("Explanation."))])
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
            plan_text="Apply this to Intervals.icu:\nProposed changes:\n  - create Tempo Session",
            draft_id=draft_id,
        )
    )
    import asyncio

    asyncio.run(cli_chat._handle_proposal(engine, state, "what is this?", session))
    user_message = provider.calls[0]["messages"][1].content
    assert "current_proposal" not in user_message
    assert provider.calls[0]["json_mode"] is True


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
    state = ChatState(plan=PlanSnapshot(plan_text="plan", draft_id=draft_id))
    session = ChatSession()
    session.context = CoachContext(focus="f", max_tokens=100)
    result = await cli_chat._handle_proposal(engine, state, "make it easier", session)
    assert isinstance(result, ChatState)
    assert result.plan is not None
    assert [row.content for row in store.list_feedback(draft_id)] == ["make it easier"]
    draft = store.get_draft(draft_id)
    assert draft is not None
    assert draft.context.current_proposal is None


def test_chat_apply_failure_after_yes_shows_retry_hint(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider, calendar=calendar)

    async def broken_apply(decision_id: int | None = None) -> Any:
        raise RuntimeError("writer exploded")

    import asyncio

    async def noop() -> None:
        pass

    engine.apply = broken_apply
    result = runner.invoke(cli_main.app, [], input="analyze my week\nyes\n")
    assert result.exit_code == 0
    assert "writer exploded" in result.output
    assert "not applied" in result.output
    assert 'say "retry"' in result.output
    decision = store.get_decision(1)
    assert decision is not None
    assert decision.applied_at is None
    asyncio.run(noop())


def test_chat_retry_applies_the_recorded_decision(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider, calendar=calendar)
    original = engine.apply
    attempts: list[int] = []

    async def flaky_apply(decision_id: int | None = None) -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("writer exploded")
        return await original(decision_id)

    engine.apply = flaky_apply
    result = runner.invoke(cli_main.app, [], input="analyze my week\nyes\nretry\n")
    assert result.exit_code == 0
    assert 'say "retry"' in result.output
    assert len(attempts) == 2
    assert len(calendar.created) == 1
    assert store.list_unapplied_decisions() == []


def test_chat_sqlite_error_survives_repl(patched: Any) -> None:
    import sqlite3

    provider = FakeLlmProvider([completion(report_json()), completion(report_json())])
    engine, _ = patched(provider)

    async def broken_analyze(focus: str, **kwargs: Any) -> Any:
        raise sqlite3.OperationalError("database is locked")

    engine.analyze = broken_analyze
    result = runner.invoke(cli_main.app, [], input="how was my week?\nand today?\n/help\n")
    assert result.exit_code == 0
    assert "error:" in result.output
    assert "/provider" in result.output


def test_chat_help_during_confirmation_skips_llm(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, [], input="analyze my week\n/help\ncancel\n")
    assert result.exit_code == 0
    assert "/provider" in result.output
    assert len(provider.calls) == 1
    assert store.list_feedback(1) == []


def test_chat_forget_during_confirmation_is_refused_without_llm(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, [], input="analyze my week\n/forget\ncancel\n")
    assert result.exit_code == 0
    assert "unavailable while a proposal is open" in result.output
    assert len(provider.calls) == 1
    assert store.list_feedback(1) == []


def test_chat_question_after_refused_forget_mid_gate_still_answered(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Explanation.")),
        ]
    )
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="analyze my week\n/forget\nwhat does this train?\nno\n"
    )
    assert result.exit_code == 0
    assert "/forget is unavailable while a proposal is open." in result.output
    assert "Coach: Explanation." in result.output
    assert len(provider.calls) == 2
    assert store.list_feedback(1) == []


def test_chat_unknown_command_during_confirmation_skips_llm(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, [], input="analyze my week\n/bogus\ncancel\n")
    assert result.exit_code == 0
    assert "Unknown command." in result.output
    assert len(provider.calls) == 1
    assert store.list_feedback(1) == []


def test_chat_leading_trend_question_refreshes_deep_analysis(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion(report_json())])
    _, store = patched(provider)
    result = runner.invoke(
        cli_main.app,
        [],
        input=(
            "how was my week?\nhow much did my heart rate improve on hills over the last 3 months\n"
        ),
    )
    assert result.exit_code == 0
    assert '"activity_detail": null' not in provider.calls[1]["messages"][1].content
    assert len(store.list_drafts()) == 2


def test_chat_discussion_does_not_open_proposal(patched: Any) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(intent="chat", mutations=[CREATE_MUTATION]))]
    )
    patched(provider)
    result = runner.invoke(cli_main.app, [], input="review my last two runs\n/exit\n")
    assert result.exit_code == 0
    assert "Confirm? Reply with exactly yes or no" not in result.output
    assert "did not read this as a planning request" in result.output
    assert len(provider.calls) == 1


def test_chat_planning_phrase_opens_proposal(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="could we plan a rest run after my rest day\ncancel\n"
    )
    assert result.exit_code == 0
    assert "Confirm? Reply with exactly yes or no" in result.output


def test_chat_analysis_with_mutations_opens_proposal(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    patched(provider)
    result = runner.invoke(cli_main.app, [], input="analyze my week\ncancel\n")
    assert result.exit_code == 0
    assert "Confirm? Reply with exactly yes or no" in result.output


def test_chat_analysis_intent_with_mutations_shows_the_refusal(patched: Any) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(intent="analysis", mutations=[CREATE_MUTATION]))]
    )
    patched(provider)
    result = runner.invoke(cli_main.app, [], input="how was my week?\n/exit\n")
    assert result.exit_code == 0
    assert "did not read this as a planning request" in result.output
    assert "Confirm? Reply with exactly yes or no" not in result.output
    assert len(provider.calls) == 1


def test_chat_material_questions_block_a_workout_plan(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(
                report_json(
                    mutations=[CREATE_MUTATION],
                    needs_input=["What is your expected finish time for the race?"],
                )
            )
        ]
    )
    patched(provider)
    result = runner.invoke(cli_main.app, [], input="plan my race\n/exit\n")
    assert result.exit_code == 0
    assert "needs answers before proposing calendar changes" in result.output
    assert "expected finish time" in result.output
    assert "proceed with assumptions" in result.output
    assert "Confirm? Reply with exactly yes or no" not in result.output


def test_chat_answer_after_needs_input_reruns_the_analysis(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(needs_input=["What is your expected finish time?"])),
            completion(report_json(mutations=[CREATE_MUTATION])),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app,
        [],
        input="plan my race\ntarget 60 minutes, I can train daily\ncancel\n",
    )
    assert result.exit_code == 0
    assert len(provider.calls) == 2
    assert provider.calls[1]["json_mode"] is True
    assert "Confirm? Reply with exactly yes or no" in result.output


def test_chat_duplicate_questions_are_shown_once(patched: Any) -> None:
    question = "What is your expected finish time?"
    provider = FakeLlmProvider(
        [
            completion(
                report_json(
                    needs_input=[question],
                    questions=[question, "Do you want cycling kept in the taper?"],
                )
            )
        ]
    )
    patched(provider)
    result = runner.invoke(cli_main.app, [], input="plan my race\n/exit\n")
    assert result.exit_code == 0
    assert result.output.count(question) == 1
    assert "Do you want cycling kept in the taper?" in result.output


def test_chat_non_plan_report_with_needs_input_shows_only_the_intent_hint(
    patched: Any,
) -> None:
    provider = FakeLlmProvider(
        [
            completion(
                report_json(
                    intent="chat",
                    mutations=[CREATE_MUTATION],
                    needs_input=["What is your expected finish time?"],
                )
            )
        ]
    )
    patched(provider)
    result = runner.invoke(cli_main.app, [], input="how was my week?\n/exit\n")
    assert result.exit_code == 0
    assert "did not read this as a planning request" in result.output
    assert "needs answers before proposing calendar changes" not in result.output


def test_chat_needs_input_without_mutations_shows_the_questions(patched: Any) -> None:
    provider = FakeLlmProvider(
        [completion(report_json(needs_input=["What is your expected finish time?"]))]
    )
    patched(provider)
    result = runner.invoke(cli_main.app, [], input="plan my race\n/exit\n")
    assert result.exit_code == 0
    assert "needs answers before proposing calendar changes" in result.output
    assert "expected finish time" in result.output
    assert "Confirm? Reply with exactly yes or no" not in result.output
    assert "Answer my questions here if you like" not in result.output


def test_chat_proceed_with_assumptions_opens_the_proposal(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(
                report_json(
                    mutations=[CREATE_MUTATION],
                    needs_input=["What is your expected finish time for the race?"],
                )
            )
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="plan my race, proceed with assumptions\ncancel\n"
    )
    assert result.exit_code == 0
    assert "Confirm? Reply with exactly yes or no" in result.output


def test_chat_bare_command_prints_a_slash_hint(patched: Any) -> None:
    provider = FakeLlmProvider()
    patched(provider)
    result = runner.invoke(cli_main.app, [], input="forget 7\n/exit\n")
    assert result.exit_code == 0
    assert "type it with a slash" in result.output
    assert provider.calls == []


def test_chat_unexpected_error_keeps_the_session(patched: Any) -> None:
    provider = FakeLlmProvider()
    engine, _ = patched(provider)

    async def broken_analyze(focus: str, **kwargs: Any) -> Any:
        raise KeyError("boom")

    engine.analyze = broken_analyze
    result = runner.invoke(cli_main.app, [], input="how was my week?\n/exit\n")
    assert result.exit_code == 0
    assert "error:" in result.output
    assert "bye" in result.output


def test_chat_forget_with_days_keeps_recent_history(patched: Any) -> None:
    provider = FakeLlmProvider([completion(report_json()), completion(report_json())])
    _, store = patched(provider)
    draft_id = store.save_draft(
        focus="f",
        report=DecisionReport.model_validate(json.loads(report_json())),
        context=CoachContext(focus="f"),
    )
    store.add_feedback(draft_id, "legs heavy")
    result = runner.invoke(cli_main.app, [], input="/forget 30\nhow was my week?\n")
    assert result.exit_code == 0
    assert "history older than 30 days" in result.output
    assert "legs heavy" in provider.calls[0]["messages"][1].content


def test_chat_startup_reports_pruned_records(
    patched: Any, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    provider = FakeLlmProvider([completion(report_json())])
    engine, _ = patched(provider)
    monkeypatch.setattr(engine, "prune_history", lambda days=None, **kwargs: {"drafts": 2})
    monkeypatch.setattr(
        cli_main, "get_settings", lambda: settings.model_copy(update={"history_days": 180})
    )
    result = runner.invoke(cli_main.app, [], input="how was my week?\n")
    assert result.exit_code == 0
    assert "Pruned 2 old records" in result.output



def test_chat_negated_assume_does_not_override_needs_input(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(
                report_json(mutations=[CREATE_MUTATION], needs_input=["What is your goal time?"])
            )
        ]
    )
    patched(provider)
    result = runner.invoke(cli_main.app, [], input="plan my block - don't assume anything\n/exit\n")
    assert result.exit_code == 0
    assert "needs answers before proposing calendar changes" in result.output
    assert "Confirm? Reply with exactly yes or no" not in result.output


def test_chat_negated_use_assumptions_does_not_override(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(
                report_json(mutations=[CREATE_MUTATION], needs_input=["What is your goal time?"])
            )
        ]
    )
    patched(provider)
    result = runner.invoke(cli_main.app, [], input="plan my block - don't use assumptions\n/exit\n")
    assert result.exit_code == 0
    assert "needs answers before proposing calendar changes" in result.output
    assert "Confirm? Reply with exactly yes or no" not in result.output


def test_chat_revision_with_needs_input_keeps_the_gate_closed(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(
                report_json("Need more.", mutations=[CREATE_MUTATION], needs_input=["Goal time?"])
            ),
        ]
    )
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, [], input="analyze my week\nmake it easier\n/exit\n")
    assert result.exit_code == 0
    assert "needs answers before proposing calendar changes" in result.output
    assert [row.content for row in store.list_feedback(1)] == ["make it easier"]


def test_chat_revision_to_chat_intent_closes_the_gate(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Just advice.", intent="chat", mutations=[CREATE_MUTATION])),
        ]
    )
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, [], input="analyze my week\nmake it easier\n/exit\n")
    assert result.exit_code == 0
    assert "did not read that as a planning request" in result.output
    assert [row.content for row in store.list_feedback(1)] == ["make it easier"]
    assert result.output.count("Confirm? Reply with exactly yes or no") == 1


def test_chat_question_first_change_request_is_answered_and_gated(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Revised.", mutations=[CREATE_MUTATION])),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="analyze my week\nhow about 45 minutes instead?\nno\n"
    )
    assert result.exit_code == 0
    assert len(provider.calls) == 2
    prompt = provider.calls[1]["messages"][1].content
    assert "Current message:\nhow about 45 minutes instead?" in prompt
    assert result.output.count("Confirm? Reply with exactly yes or no") == 2


def test_chat_retry_with_nothing_pending_does_not_write(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider()
    engine, store = patched(provider, calendar=calendar)

    async def broken_apply(decision_id: int | None = None) -> Any:
        raise AssertionError("apply must not be called")

    engine.apply = broken_apply
    result = runner.invoke(cli_main.app, [], input="retry\n/exit\n")
    assert result.exit_code == 0
    assert "Nothing to apply." in result.output
    assert "recorded but never applied" not in result.output
    assert calendar.created == []
    assert store.list_unapplied_decisions() == []


def test_chat_startup_offers_an_unapplied_decision_after_restart(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    engine, store = patched(provider, calendar=calendar)
    original = engine.apply

    async def broken_apply(decision_id: int | None = None) -> Any:
        raise RuntimeError("writer exploded")

    engine.apply = broken_apply
    first = runner.invoke(cli_main.app, [], input="analyze my week\nyes\n/exit\n")
    assert first.exit_code == 0
    assert len(store.list_unapplied_decisions()) == 1

    engine.apply = original
    second = runner.invoke(cli_main.app, [], input="retry\n/exit\n")
    assert second.exit_code == 0
    assert "was recorded but never applied" in second.output
    assert len(calendar.created) == 1
    assert store.list_unapplied_decisions() == []


def test_chat_gate_question_with_needs_input_keeps_the_gate_closed(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(
                report_json(
                    "Need more.",
                    mutations=[CREATE_MUTATION],
                    needs_input=["What is your goal time?"],
                )
            ),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="analyze my week\nwhat does this train exactly?\nno\n"
    )
    assert result.exit_code == 0
    assert "needs answers before proposing calendar changes" in result.output
    assert "What is your goal time?" in result.output
    assert result.output.count("Confirm? Reply with exactly yes or no") == 1


def test_chat_retry_applies_each_unapplied_decision_in_order(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider()
    _, store = patched(provider, calendar=calendar)
    for index in range(2):
        mutation = dict(CREATE_MUTATION, name=f"Session {index}")
        draft_id = store.save_draft(
            focus=f"f{index}",
            report=DecisionReport.model_validate(json.loads(report_json(mutations=[mutation]))),
            context=CoachContext(focus=f"f{index}"),
        )
        store.approve_draft(draft_id)

    result = runner.invoke(cli_main.app, [], input="retry\nretry\n/exit\n")
    assert result.exit_code == 0
    assert "was recorded but never applied" in result.output
    assert "Decision #2 is still unapplied" in result.output
    assert len(calendar.created) == 2
    assert store.list_unapplied_decisions() == []


def test_chat_forget_clears_the_retry_pointer(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider()
    _, store = patched(provider, calendar=calendar)
    draft_id = store.save_draft(
        focus="f",
        report=DecisionReport.model_validate(json.loads(report_json(mutations=[CREATE_MUTATION]))),
        context=CoachContext(focus="f"),
    )
    store.approve_draft(draft_id)

    result = runner.invoke(cli_main.app, [], input="/forget\nretry\n/exit\n")
    assert result.exit_code == 0
    assert "Forgot" in result.output
    assert "Nothing to apply." in result.output
    assert "decision not found" not in result.output
    assert calendar.created == []


def test_chat_race_proposal_yes_writes_the_race(patched: Any) -> None:
    race = {
        "action": "create_race",
        "name": "Autumn Trail Race",
        "start_date_local": (date.today() + timedelta(days=30)).isoformat(),
        "category": "RACE_A",
        "type": "Run",
        "moving_time": 4200,
        "icu_training_load": 90,
    }
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[race]))])
    _, store = patched(provider, calendar=calendar)
    result = runner.invoke(cli_main.app, [], input="plan my race\nyes\n")
    assert result.exit_code == 0
    assert calendar.created[0]["category"] == "RACE_A"
    assert calendar.created[0]["moving_time"] == 4200
    assert store.list_unapplied_decisions() == []


def test_chat_race_needs_input_blocks_the_proposal(patched: Any) -> None:
    calendar = FakeCalendarClient()
    race = {
        "action": "create_race",
        "name": "Autumn Trail Race",
        "start_date_local": (date.today() + timedelta(days=30)).isoformat(),
        "category": "RACE_A",
        "type": "Run",
        "moving_time": 4200,
        "icu_training_load": 90,
    }
    provider = FakeLlmProvider(
        [
            completion(
                report_json(mutations=[race], needs_input=["What is your expected finish time?"])
            )
        ]
    )
    patched(provider, calendar=calendar)
    result = runner.invoke(cli_main.app, [], input="plan my race\n/exit\n")
    assert result.exit_code == 0
    assert "needs answers before proposing calendar changes" in result.output
    assert "Confirm? Reply with exactly yes or no" not in result.output
    assert calendar.created == []


def test_chat_startup_discards_a_stale_approved_decision(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider()
    _, store = patched(provider, calendar=calendar)
    today = datetime.now(ZoneInfo("Europe/Paris")).date()
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

    result = runner.invoke(cli_main.app, [], input="retry\n/exit\n")
    assert result.exit_code == 0
    assert "was approved with dates that have passed; discarded" in result.output
    assert "was recorded but never applied" not in result.output
    assert "Nothing to apply." in result.output
    assert store.list_unapplied_decisions() == []


def test_retry_apply_discards_a_stale_decision(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider()
    engine, store = patched(provider, calendar=calendar)
    today = datetime.now(ZoneInfo("Europe/Paris")).date()
    report = DecisionReport.model_validate(
        json.loads(
            report_json(
                mutations=[
                    {
                        "action": "create",
                        "name": "Tomorrow Session",
                        "start_date_local": (today + timedelta(days=1)).isoformat(),
                        "moving_time": 3600,
                    }
                ]
            )
        )
    )
    draft_id = store.save_draft(focus="f", report=report, context=CoachContext(focus="f"))
    decision = store.approve_draft(draft_id)
    session = ChatSession(cap=100)
    session.pending_decision_id = decision.id
    engine.today = lambda: today + timedelta(days=2)

    asyncio.run(cli_chat._retry_apply(engine, session, "retry"))

    assert session.pending_decision_id is None
    assert store.list_unapplied_decisions() == []
    assert calendar.created == []


def test_chat_non_exact_yes_is_feedback_and_writes_nothing(patched: Any) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Revised.", mutations=[CREATE_MUTATION])),
        ]
    )
    patched(provider, calendar=calendar)
    result = runner.invoke(cli_main.app, [], input="analyze my week\n1 hour, yes please\n/exit\n")
    assert result.exit_code == 0
    assert "Nothing was written" in result.output
    assert 'reply exactly "yes"' in result.output
    assert calendar.created == []
    assert len(provider.calls) == 2
    assert "Recent conversation:" in provider.calls[1]["messages"][1].content


def test_chat_forget_rejects_invalid_day_counts(patched: Any) -> None:
    provider = FakeLlmProvider()
    _, store = patched(provider)
    result = runner.invoke(cli_main.app, [], input="/forget 0\n/forget abc\n/exit\n")
    assert result.exit_code == 0
    assert result.output.count("Usage: /forget [days>0]") == 2
    assert "Forgot" not in result.output
    assert store.list_drafts() == []


def test_chat_question_with_chat_intent_does_not_open_a_gate(patched: Any) -> None:
    provider = FakeLlmProvider(
        [
            completion(report_json(mutations=[CREATE_MUTATION])),
            completion(report_json("Answer.", intent="chat", mutations=[CREATE_MUTATION])),
        ]
    )
    patched(provider)
    result = runner.invoke(
        cli_main.app, [], input="analyze my week\nwhat does this train exactly?\n/exit\n"
    )
    assert result.exit_code == 0
    assert "did not read that as a planning request" in result.output
    assert result.output.count("Confirm? Reply with exactly yes or no") == 2


def test_chat_render_failure_after_apply_is_not_reported_as_unapplied(
    patched: Any, monkeypatch: Any
) -> None:
    calendar = FakeCalendarClient()
    provider = FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))])
    patched(provider, calendar=calendar)

    def boom(report: Any) -> None:
        raise RuntimeError("render exploded")

    monkeypatch.setattr(cli_chat, "render_apply", boom)
    result = runner.invoke(cli_main.app, [], input="analyze my week\nyes\n/exit\n")
    assert result.exit_code == 0
    assert len(calendar.created) == 1
    assert "recorded but not applied" not in result.output

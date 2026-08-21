from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from open_endurance_coach.cli import main as cli_main
from open_endurance_coach.config import Settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import DraftStatus

from .fakes import FakeCalendarClient, FakeLlmProvider, completion, report_json
from .test_cli import CREATE_MUTATION, FakeRunner, make_engine

runner = CliRunner()


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path) -> Any:
    def build(
        provider: FakeLlmProvider, calendar: FakeCalendarClient | None = None
    ) -> tuple[CoachEngine, CoachStore]:
        engine, store = make_engine(settings, tmp_path, provider, calendar=calendar)
        monkeypatch.setattr(cli_main, "_with_engine", FakeRunner(engine))
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
    patched(FakeLlmProvider())
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


def test_chat_approve_is_not_wired_and_writes_nothing(patched: Any) -> None:
    calendar = FakeCalendarClient()
    _, store = patched(
        FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]),
        calendar=calendar,
    )
    result = runner.invoke(
        cli_main.app, ["chat"], input="/analyze\n/approve 1\n/apply 1 --write\n/exit\n"
    )
    assert result.exit_code == 0
    assert "2b" in result.output
    draft = store.get_draft(1)
    assert draft is not None
    assert draft.status is DraftStatus.PENDING
    assert store.list_decisions() == []
    assert calendar.created == []


def test_chat_converse_turns_are_not_wired_yet(patched: Any) -> None:
    _, store = patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="how was my week?\n")
    assert result.exit_code == 0
    assert "4c" in result.output
    assert store.list_drafts() == []


def test_chat_blank_lines_are_skipped(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["chat"], input="\n   \n/help\n")
    assert result.exit_code == 0
    assert "error" not in result.output
    assert "/analyze" in result.output

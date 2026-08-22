import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from open_endurance_coach.cli import main as cli_main
from open_endurance_coach.clients.llm import LlmClient
from open_endurance_coach.config import Settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import DraftStatus
from open_endurance_coach.writer.calendar import CalendarWriter

from .fakes import (
    FakeCalendarClient,
    FakeLlmProvider,
    RecordingSleep,
    completion,
    make_event,
    make_intervals_client,
    report_json,
)

runner = CliRunner()

CREATE_MUTATION = {
    "action": "create",
    "name": "Tempo Session",
    "start_date_local": "2024-02-05",
    "moving_time": 3600,
}


def make_engine(
    settings: Settings,
    tmp_path: Path,
    provider: FakeLlmProvider,
    calendar: FakeCalendarClient | None = None,
) -> tuple[CoachEngine, CoachStore]:
    llm = LlmClient(
        settings.model_copy(update={"llm_provider": "fake"}),
        {"fake": provider},
        sleep=RecordingSleep(),
    )
    store = CoachStore(tmp_path / "coach.db")
    writer = CalendarWriter(calendar) if calendar is not None else None
    engine = CoachEngine(settings, store, make_intervals_client(), llm, writer=writer)
    return engine, store


class FakeRunner:
    def __init__(self, engine: CoachEngine) -> None:
        self.engine = engine

    async def __call__(self, callback: Callable[[CoachEngine], Awaitable[None]]) -> None:
        await callback(self.engine)


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path) -> Any:
    def build(
        provider: FakeLlmProvider, calendar: FakeCalendarClient | None = None
    ) -> tuple[CoachEngine, CoachStore]:
        engine, store = make_engine(settings, tmp_path, provider, calendar=calendar)
        monkeypatch.setattr(cli_main, "_with_engine", FakeRunner(engine))
        return engine, store

    return build


def test_ask_prints_summary_and_draft_id(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["ask", "How is my form?"])
    assert result.exit_code == 0
    assert "Load stable." in result.output
    assert "Draft #1" in result.output


def test_analyze_default_focus_creates_draft(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["analyze"])
    assert result.exit_code == 0
    assert "Draft #1" in result.output


def test_analyze_feedback_flag_injected(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    result = runner.invoke(cli_main.app, ["analyze", "--feedback", "legs heavy"])
    assert result.exit_code == 0
    assert store.list_drafts()[0].user_feedback == "legs heavy"


def test_review_lists_pending_drafts(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json()), completion(report_json("Week two."))]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["review"])
    assert result.exit_code == 0
    assert "Draft #2" in result.output
    assert "Draft #1" in result.output


def test_review_shows_solicitations(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["review", "1"])
    assert result.exit_code == 0
    assert "RPE missing" in result.output
    assert "Fueling" in result.output


def test_review_missing_draft_fails_gracefully(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["review", "404"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_approve_records_decision(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["approve", "1", "--yes"])
    assert result.exit_code == 0
    assert "Decision #1" in result.output
    assert store.get_draft(1) is not None
    assert store.get_draft(1).status is DraftStatus.APPROVED
    assert len(store.list_decisions()) == 1


def test_approve_with_mutations_file(patched: Any, tmp_path: Path) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    mutations_path = tmp_path / "mutations.json"
    mutations_path.write_text(
        json.dumps(
            [{"action": "create", "name": "Custom Session", "start_date_local": "2024-02-06"}]
        )
    )
    result = runner.invoke(
        cli_main.app, ["approve", "1", "--mutations-file", str(mutations_path), "--yes"]
    )
    assert result.exit_code == 0
    decision = store.list_decisions()[0]
    assert decision.report.mutations[0].name == "Custom Session"


def test_approve_invalid_mutations_file_fails(patched: Any, tmp_path: Path) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    mutations_path = tmp_path / "mutations.json"
    mutations_path.write_text("not json")
    result = runner.invoke(
        cli_main.app, ["approve", "1", "--mutations-file", str(mutations_path), "--yes"]
    )
    assert result.exit_code == 2
    assert "invalid mutations file" in result.output


def test_approve_missing_mutations_file_fails(patched: Any, tmp_path: Path) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(
        cli_main.app, ["approve", "1", "--mutations-file", str(tmp_path / "missing.json"), "--yes"]
    )
    assert result.exit_code == 2
    assert "invalid mutations file" in result.output


def test_reject_flips_status(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["reject", "1", "--yes"])
    assert result.exit_code == 0
    assert "rejected" in result.output
    assert store.get_draft(1).status is DraftStatus.REJECTED


def test_ask_llm_failure_exits_with_error(patched: Any) -> None:
    patched(FakeLlmProvider([completion(""), completion(""), completion("")]))
    result = runner.invoke(cli_main.app, ["ask", "How is my form?"])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_feedback_updates_draft_and_prints_revised_summary(patched: Any) -> None:
    _, store = patched(
        FakeLlmProvider(
            [completion(report_json()), completion(report_json("Revised after feedback."))]
        )
    )
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["feedback", "1", "Legs heavy, RPE 8"])
    assert result.exit_code == 0
    assert "Revised after feedback." in result.output
    assert "Draft #1 updated" in result.output
    draft = store.get_draft(1)
    assert draft is not None
    assert draft.user_feedback == "Legs heavy, RPE 8"
    assert draft.report.summary == "Revised after feedback."
    assert draft.status is DraftStatus.PENDING


def test_feedback_missing_draft_fails_gracefully(patched: Any) -> None:
    patched(FakeLlmProvider())
    result = runner.invoke(cli_main.app, ["feedback", "404", "RPE was 7"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_review_suggests_feedback_command(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["review", "1"])
    assert result.exit_code == 0
    assert "RPE missing" in result.output
    assert "coach feedback 1" in result.output


def test_review_no_suggestion_after_approve(patched: Any) -> None:
    patched(FakeLlmProvider([completion(report_json())]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["approve", "1", "--yes"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["review", "1"])
    assert result.exit_code == 0
    assert "coach feedback 1" not in result.output


def decision_of(store: CoachStore, decision_id: int) -> Any:
    decision = store.get_decision(decision_id)
    assert decision is not None
    return decision


def approve_draft(patched: Any) -> tuple[FakeCalendarClient, CoachStore]:
    calendar = FakeCalendarClient()
    _, store = patched(
        FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]),
        calendar=calendar,
    )
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["approve", "1", "--yes"], catch_exceptions=False)
    return calendar, store


def test_apply_dry_run_by_default(patched: Any) -> None:
    calendar, store = approve_draft(patched)
    result = runner.invoke(cli_main.app, ["apply"])
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert "create" in result.output
    assert calendar.created == []
    assert decision_of(store, 1).applied_at is None


def test_apply_write_flag_writes(patched: Any) -> None:
    calendar, store = approve_draft(patched)
    result = runner.invoke(cli_main.app, ["apply", "--write", "--yes"])
    assert result.exit_code == 0
    assert "created" in result.output
    assert "DRY RUN" not in result.output
    assert len(calendar.created) == 1
    assert decision_of(store, 1).applied_at is not None


def test_apply_specific_decision_only(patched: Any) -> None:
    calendar = FakeCalendarClient()
    _, store = patched(
        FakeLlmProvider(
            [
                completion(report_json()),
                completion(report_json(mutations=[CREATE_MUTATION])),
            ]
        ),
        calendar=calendar,
    )
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["approve", "1", "--yes"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["approve", "2", "--yes"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["apply", "2"])
    assert result.exit_code == 0
    assert "Decision #2" in result.output
    assert "Decision #1" not in result.output
    assert calendar.created == []
    assert decision_of(store, 1).applied_at is None


def test_apply_no_unapplied_decisions(patched: Any) -> None:
    patched(FakeLlmProvider(), calendar=FakeCalendarClient())
    result = runner.invoke(cli_main.app, ["apply"])
    assert result.exit_code == 0
    assert "No unapplied decisions" in result.output


def test_apply_error_propagates(patched: Any) -> None:
    calendar = FakeCalendarClient([make_event(10001, "2024-02-05", category="RACE_B")])
    _, store = patched(
        FakeLlmProvider(
            [
                completion(
                    report_json(
                        mutations=[{"action": "update", "event_id": 10001, "moving_time": 4200}]
                    )
                )
            ]
        ),
        calendar=calendar,
    )
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["approve", "1", "--yes"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["apply", "--write", "--yes"])
    assert result.exit_code == 1
    assert "non-WORKOUT" in result.output
    assert decision_of(store, 1).applied_at is None


def test_missing_env_shows_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_settings() -> Settings:
        return Settings(_env_file=None)

    monkeypatch.setattr(cli_main, "get_settings", broken_settings)
    result = runner.invoke(cli_main.app, ["review"])
    assert result.exit_code == 1
    assert "configuration missing" in result.output
    assert ".env" in result.output


def test_approve_gate_proceeds_on_yes(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["approve", "1"], input="yes\n")
    assert result.exit_code == 0
    assert "Confirm?" in result.output
    assert "Decision #1 recorded" in result.output
    assert store.get_draft(1).status is DraftStatus.APPROVED


def test_approve_gate_no_declines(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["approve", "1"], input="no\n")
    assert result.exit_code == 0
    assert "Nothing changed." in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_approve_gate_eof_aborts_without_writing(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["approve", "1"])
    assert result.exit_code == 0
    assert "Cancelled. Nothing changed." in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []


def test_approve_gate_blank_line_re_prompts(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["approve", "1"], input="\nyes\n")
    assert result.exit_code == 0
    assert result.output.count("Confirm?") == 2
    assert "Decision #1 recorded" in result.output
    assert store.get_draft(1).status is DraftStatus.APPROVED


def test_approve_gate_feedback_fallback_restates_plan(patched: Any) -> None:
    _, store = patched(
        FakeLlmProvider(
            [
                completion(report_json(mutations=[CREATE_MUTATION])),
                completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
            ]
        )
    )
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["approve", "1"], input="wait, explain\nyes\n")
    assert result.exit_code == 0
    assert [row.content for row in store.list_feedback(1)] == ["wait, explain"]
    assert result.output.count("Proposed changes:") == 2
    assert "Decision #1 recorded" in result.output


def test_approve_gate_restates_override_plan_after_fallback(patched: Any, tmp_path: Path) -> None:
    _, store = patched(
        FakeLlmProvider(
            [
                completion(report_json(mutations=[CREATE_MUTATION])),
                completion(report_json("Reconsidered.", mutations=[CREATE_MUTATION])),
            ]
        )
    )
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    mutations_path = tmp_path / "mutations.json"
    mutations_path.write_text(
        json.dumps(
            [{"action": "create", "name": "Custom Session", "start_date_local": "2024-02-06"}]
        )
    )
    result = runner.invoke(
        cli_main.app,
        ["approve", "1", "--mutations-file", str(mutations_path)],
        input="hmm\nyes\n",
    )
    assert result.exit_code == 0
    assert "Custom Session" in result.output
    assert "Tempo Session" not in result.output
    decision = store.list_decisions()[0]
    assert decision.report.mutations[0].name == "Custom Session"


def test_approve_gate_rejects_non_pending_before_prompt(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["approve", "1", "--yes"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["approve", "1"])
    assert result.exit_code == 1
    assert "only pending drafts can be approved" in result.output
    assert "Confirm?" not in result.output
    assert store.get_draft(1).status is DraftStatus.APPROVED


def test_reject_gate_proceeds_on_yes(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["reject", "1"], input="yes\n")
    assert result.exit_code == 0
    assert "Draft #1 rejected." in result.output
    assert store.get_draft(1).status is DraftStatus.REJECTED


def test_reject_gate_no_keeps_draft_pending(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json())]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["reject", "1"], input="no\n")
    assert result.exit_code == 0
    assert "Nothing changed." in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING


def test_apply_write_gate_proceeds_on_yes(patched: Any) -> None:
    calendar = FakeCalendarClient()
    _, store = patched(
        FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]),
        calendar=calendar,
    )
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["approve", "1", "--yes"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["apply", "--write"], input="yes\n")
    assert result.exit_code == 0
    assert "Confirm?" in result.output
    assert len(calendar.created) == 1
    assert decision_of(store, 1).applied_at is not None


def test_apply_write_gate_no_writes_nothing(patched: Any) -> None:
    calendar = FakeCalendarClient()
    _, store = patched(
        FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]),
        calendar=calendar,
    )
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["approve", "1", "--yes"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["apply", "--write"], input="no\n")
    assert result.exit_code == 0
    assert "Nothing changed." in result.output
    assert calendar.created == []
    assert decision_of(store, 1).applied_at is None


def test_apply_write_gate_discuss_re_prompts_same_plan(patched: Any) -> None:
    calendar = FakeCalendarClient()
    _, store = patched(
        FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]),
        calendar=calendar,
    )
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    runner.invoke(cli_main.app, ["approve", "1", "--yes"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["apply", "--write"], input="explain first?\nyes\n")
    assert result.exit_code == 0
    assert result.output.count("Confirm?") == 2
    assert "coach chat" in result.output
    assert len(calendar.created) == 1
    assert decision_of(store, 1).applied_at is not None


def test_approve_gate_exit_command_cancels(patched: Any) -> None:
    _, store = patched(FakeLlmProvider([completion(report_json(mutations=[CREATE_MUTATION]))]))
    runner.invoke(cli_main.app, ["analyze"], catch_exceptions=False)
    result = runner.invoke(cli_main.app, ["approve", "1"], input="/exit\n")
    assert result.exit_code == 0
    assert "Cancelled. Nothing changed." in result.output
    assert store.get_draft(1).status is DraftStatus.PENDING
    assert store.list_decisions() == []

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from open_endurance_coach.cli.rendering import (
    render_apply,
    render_draft,
    render_report,
    render_review,
)
from open_endurance_coach.engine.coach import ReviewView
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport
from open_endurance_coach.store.records import Draft, DraftStatus
from open_endurance_coach.writer.records import AppliedDecision, ApplyReport, MutationOutcome

from .fakes import report_json


def make_draft(summary: str = "Load stable.") -> Draft:
    return Draft(
        id=1,
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        status=DraftStatus.PENDING,
        focus="focus",
        user_feedback=None,
        context=CoachContext(focus="focus"),
        report=DecisionReport.model_validate(json.loads(report_json(summary))),
    )


def test_render_report_prints_summary_findings_questions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    render_report(make_draft().report)
    out = capsys.readouterr().out
    assert "Coach: Load stable." in out
    assert "- Tempo block hit target." in out
    assert "? RPE on Thursday?" in out


def test_render_draft_one_shot_hint(capsys: pytest.CaptureFixture[str]) -> None:
    render_draft(make_draft())
    out = capsys.readouterr().out
    assert "Draft #1 saved (pending). Review it: coach review 1" in out


def test_render_draft_updated_verb(capsys: pytest.CaptureFixture[str]) -> None:
    render_draft(make_draft(), updated=True)
    out = capsys.readouterr().out
    assert "Draft #1 updated (pending)" in out


def test_render_draft_chat_hint(capsys: pytest.CaptureFixture[str]) -> None:
    render_draft(make_draft(), chat=True)
    out = capsys.readouterr().out
    assert "Review it: /review 1" in out
    assert "coach review 1" not in out


def test_render_review_one_shot_hint(capsys: pytest.CaptureFixture[str]) -> None:
    view = ReviewView(draft=make_draft(), requested_feedback=["RPE missing for Ride:"])
    render_review(view)
    out = capsys.readouterr().out
    assert "? RPE missing for Ride:" in out
    assert 'Answer the coach: coach feedback 1 "your RPE and notes"' in out


def test_render_review_chat_hint(capsys: pytest.CaptureFixture[str]) -> None:
    view = ReviewView(draft=make_draft(), requested_feedback=["RPE missing for Ride:"])
    render_review(view, chat=True)
    out = capsys.readouterr().out
    assert 'Answer the coach: /feedback 1 "your RPE and notes"' in out
    assert "coach feedback 1" not in out


def test_render_review_skips_solicitations_when_not_pending(
    capsys: pytest.CaptureFixture[str],
) -> None:
    draft = make_draft()
    approved = replace(draft, status=DraftStatus.APPROVED)
    view = ReviewView(draft=approved, requested_feedback=["RPE missing for Ride:"])
    render_review(view)
    out = capsys.readouterr().out
    assert "RPE missing" not in out
    assert "Answer the coach" not in out


def test_render_apply_dry_run_banner_and_outcomes(capsys: pytest.CaptureFixture[str]) -> None:
    report = ApplyReport(
        decisions=[
            AppliedDecision(
                decision_id=1,
                outcomes=[
                    MutationOutcome(action="create", target="Tempo Session", name="Tempo Session")
                ],
            )
        ]
    )
    render_apply(report, write=False)
    out = capsys.readouterr().out
    assert "DRY RUN - no changes written" in out
    assert "Decision #1:" in out
    assert "- create Tempo Session: Tempo Session" in out


def test_render_apply_write_mode(capsys: pytest.CaptureFixture[str]) -> None:
    report = ApplyReport(
        decisions=[
            AppliedDecision(
                decision_id=1,
                outcomes=[MutationOutcome(action="update", target="event", event_id=10001)],
            )
        ]
    )
    render_apply(report, write=True)
    out = capsys.readouterr().out
    assert "Applied:" in out
    assert "DRY RUN" not in out
    assert "- update event: 10001" in out


def test_render_apply_empty_report(capsys: pytest.CaptureFixture[str]) -> None:
    render_apply(ApplyReport(), write=False)
    out = capsys.readouterr().out
    assert "No unapplied decisions." in out

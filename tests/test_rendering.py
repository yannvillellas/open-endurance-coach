import json
from datetime import UTC, date, datetime
from typing import Any

import pytest
from rich.console import Console

from open_endurance_coach.cli.rendering import (
    apply_plan_text,
    console,
    mutations_plan_text,
    print_error,
    render_apply,
    render_report,
    split_finding_topic,
    wrap_plan_text,
)
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import (
    CreateRace,
    CreateWorkout,
    DecisionReport,
    DeleteRace,
    Mutation,
    UpdateRace,
    UpdateWorkout,
)
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
    assert "Coach" in out
    assert "Load stable." in out
    assert "Evidence" in out
    assert "- Tempo block hit target." in out
    assert "Open questions" in out
    assert "? RPE on Thursday?" in out


def test_mutations_plan_text_shows_dates_and_descriptions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import date

    from open_endurance_coach.cli.rendering import mutations_plan_text
    from open_endurance_coach.schemas.decisions import CreateWorkout, Mutation

    mutations: list[Mutation] = [
        CreateWorkout(
            action="create",
            name="Aerobic Swim",
            start_date_local=date(2026, 8, 23),
            description="2000m easy",
        ),
        CreateWorkout(action="create", name="Bare Session", start_date_local=date(2026, 8, 24)),
    ]
    text = mutations_plan_text(mutations)
    assert "Proposed changes:" in text
    assert "  2026-08-23" in text
    assert "- create Aerobic Swim: 2000m easy" in text
    assert "  2026-08-24" in text
    assert "- create Bare Session" in text


def test_render_apply_prints_outcomes(capsys: pytest.CaptureFixture[str]) -> None:
    report = ApplyReport(
        decisions=[
            AppliedDecision(
                decision_id=1,
                outcomes=[MutationOutcome(action="update", target="updated", event_id=10001)],
            )
        ]
    )
    render_apply(report)
    out = capsys.readouterr().out
    assert "Applied:" in out
    assert "DRY RUN" not in out
    assert "- update -> updated event 10001" in out


def test_render_apply_empty_report(capsys: pytest.CaptureFixture[str]) -> None:
    render_apply(ApplyReport())
    out = capsys.readouterr().out
    assert "No unapplied decisions." in out


def test_mutations_plan_text_lists_each_mutation() -> None:
    mutations: list[Mutation] = [
        CreateWorkout(
            action="create",
            name="Tempo Session",
            start_date_local=date(2024, 2, 5),
            type="Ride",
            moving_time=3600,
            icu_training_load=84,
        ),
        UpdateWorkout(action="update", event_id=10001, moving_time=4200),
    ]
    text = mutations_plan_text(mutations)
    assert "Proposed changes:" in text
    assert "  2024-02-05" in text
    assert "- create Tempo Session (Ride, moving_time=3600, load=84)" in text
    assert "(no date)" in text
    assert "- update event 10001: moving_time=4200" in text


def test_mutations_plan_text_nests_multiline_description() -> None:
    mutations: list[Mutation] = [
        CreateWorkout(
            action="create",
            name="Intervals",
            start_date_local=date(2026, 9, 19),
            description="- 10m warmup\n\n3x\n- 3m hard",
        ),
    ]
    lines = mutations_plan_text(mutations).splitlines()
    day = lines.index("  2026-09-19")
    assert lines[day + 1] == "    - create Intervals"
    assert lines[day + 2] == "      - 10m warmup"
    assert lines[day + 3] == ""
    assert lines[day + 4] == "      3x"
    assert lines[day + 5] == "      - 3m hard"


def test_mutations_plan_text_empty_mutations() -> None:
    text = mutations_plan_text([])
    assert "(no calendar changes)" in text


def test_apply_plan_text_lists_decisions_and_outcomes() -> None:
    report = ApplyReport(
        decisions=[
            AppliedDecision(
                decision_id=1,
                outcomes=[
                    MutationOutcome(action="create", target="created", name="Tempo Session"),
                    MutationOutcome(action="update", target="updated", event_id=10001),
                ],
            )
        ]
    )
    text = apply_plan_text(report)
    assert "Decision #1:" in text
    assert "- create -> created: Tempo Session" in text
    assert "- update -> updated event 10001" in text


def test_render_report_escapes_llm_markup(capsys: pytest.CaptureFixture[str]) -> None:
    from open_endurance_coach.cli.rendering import render_report

    report = DecisionReport(summary="Weird [bold]summary[/bold].")
    render_report(report)
    captured = capsys.readouterr().out
    assert "Weird [bold]summary[/bold]." in captured


def test_plan_texts_escape_llm_markup() -> None:
    from open_endurance_coach.cli.rendering import apply_plan_text, mutations_plan_text
    from open_endurance_coach.writer.records import AppliedDecision, ApplyReport, MutationOutcome

    mutations: list[Mutation] = [
        CreateWorkout(
            action="create",
            name="Weird [bold]Session[/bold]",
            start_date_local=date(2026, 8, 23),
        ),
    ]
    text = mutations_plan_text(mutations)
    assert "Weird \\[bold]Session\\[/bold]" in text
    report = ApplyReport(
        decisions=[
            AppliedDecision(
                decision_id=1,
                outcomes=[MutationOutcome(action="create", target="created", name="Weird [x]")],
            )
        ]
    )
    apply = apply_plan_text(report)
    assert "Weird \\[x]" in apply


def test_apply_plan_text_escapes_event_id(capsys: pytest.CaptureFixture[str]) -> None:
    report = ApplyReport(
        decisions=[
            AppliedDecision(
                decision_id=1,
                outcomes=[
                    MutationOutcome(action="update", target="updated", event_id="[red]9[/red]")
                ],
            )
        ]
    )
    console.print(apply_plan_text(report))
    out = capsys.readouterr().out
    assert "[red]9[/red]" in out
    assert "\\[red]9\\[/red]" not in out


def test_print_error_escapes_exception_text(capsys: pytest.CaptureFixture[str]) -> None:
    print_error(ValueError("bad [bold]payload[/bold]"))
    out = capsys.readouterr().out
    assert "error:" in out
    assert "bad [bold]payload[/bold]" in out


def test_mutations_plan_text_renders_race_create_with_type_and_category() -> None:
    mutations: list[Mutation] = [
        CreateRace(
            action="create_race",
            name="Autumn Trail Race",
            start_date_local=date(2026, 9, 27),
            category="RACE_A",
            type="TrailRun",
            moving_time=10800,
            distance=10900,
            icu_training_load=142,
            description="hilly loop",
        ),
    ]
    text = mutations_plan_text(mutations)
    assert "  2026-09-27" in text
    assert (
        "- create RACE_A Autumn Trail Race"
        " (TrailRun, moving_time=10800, distance=10900m, load=142): hilly loop" in text
    )


def test_mutations_plan_text_renders_race_update_and_delete() -> None:
    mutations: list[Mutation] = [
        UpdateRace(
            action="update_race",
            event_id=20001,
            category="RACE_B",
            type="Run",
            moving_time=7200,
        ),
        DeleteRace(action="delete_race", event_id=20002),
    ]
    text = mutations_plan_text(mutations)
    assert "- update race event 20001: category=RACE_B, type=Run, moving_time=7200" in text
    assert "- delete race event 20002" in text


def test_render_apply_warns_about_skipped_mutations(capsys: Any) -> None:
    from open_endurance_coach.cli.rendering import render_apply
    from open_endurance_coach.writer.records import AppliedDecision, ApplyReport

    report = ApplyReport(
        decisions=[AppliedDecision(decision_id=7, outcomes=[], skipped=["past-dated"])]
    )
    render_apply(report)
    out = " ".join(capsys.readouterr().out.split())
    assert "Decision #7: 1 mutation(s) skipped (past-dated)" in out
    assert "only partially updated" in out


def test_wrap_plan_text_keeps_hanging_indent() -> None:
    line = "    - create Trail Hill Sharpening (TrailRun, moving_time=2580, load=55)"
    wrapped = wrap_plan_text(line, 40).splitlines()
    assert all(len(part) <= 40 for part in wrapped)
    assert wrapped[0].startswith("    - create")
    assert all(part.startswith("      ") for part in wrapped[1:])


def test_wrap_plan_text_honours_an_explicit_hanging_indent() -> None:
    wrapped = wrap_plan_text("  " + "word " * 40, 40, hanging=2).splitlines()
    assert len(wrapped) > 1
    assert all(len(part) - len(part.lstrip(" ")) == 2 for part in wrapped)


def test_prompt_plan_frames_the_proposal(capsys: pytest.CaptureFixture[str]) -> None:
    from open_endurance_coach.chat.gate import PlanSnapshot
    from open_endurance_coach.cli.confirmation import prompt_plan

    prompt_plan(
        PlanSnapshot(
            plan_text="Apply this to Intervals.icu:\nProposed changes:\n  2026-09-17",
            draft_id=1,
        )
    )
    out = capsys.readouterr().out
    assert "Proposal" in out
    assert "Apply this to Intervals.icu" in out
    assert "Confirm? Reply exactly yes to apply" in out
    assert "(yes / no, or describe a change)" in out


def test_split_finding_topic_extracts_a_short_label() -> None:
    assert split_finding_topic("Wellness: CTL 28.99") == ("Wellness", ": CTL 28.99")
    assert split_finding_topic("Race: 11 km on 2026-09-27") == ("Race", ": 11 km on 2026-09-27")


def test_split_finding_topic_leaves_long_or_missing_prefixes_alone() -> None:
    long_prefix = "Load is run-only for three weeks: 227"
    assert split_finding_topic(long_prefix) == ("", long_prefix)
    assert split_finding_topic("no colon here") == ("", "no colon here")


def test_render_report_styles_a_finding_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    import open_endurance_coach.cli.rendering as rendering

    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=True,
        width=80,
        theme=rendering.THEME,
        color_system="standard",
        highlight=False,
    )
    monkeypatch.setattr(rendering, "console", console)
    rendering.render_report(
        DecisionReport(summary="Ok.", findings=["Wellness: CTL 28.99, form +2.37."])
    )
    assert "\x1b[1;36mWellness\x1b[0m: CTL 28.99, form +2.37." in buffer.getvalue()


def test_render_report_skips_blank_findings_and_questions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    render_report(DecisionReport(summary="Ok.", findings=["  "], questions=[""]))
    out = capsys.readouterr().out
    assert "Evidence:" not in out
    assert "Open questions:" not in out


def test_prompt_plan_does_not_render_llm_markup(capsys: pytest.CaptureFixture[str]) -> None:
    from open_endurance_coach.chat.gate import PlanSnapshot
    from open_endurance_coach.cli.confirmation import prompt_plan

    plan_text = "Apply this to Intervals.icu:\n" + mutations_plan_text(
        [
            CreateWorkout(
                action="create",
                name="Weird [bold]Session[/bold]",
                start_date_local=date(2026, 9, 17),
            )
        ]
    )
    prompt_plan(PlanSnapshot(plan_text=plan_text, draft_id=1))
    out = capsys.readouterr().out
    assert "[bold]Session[/bold]" in out

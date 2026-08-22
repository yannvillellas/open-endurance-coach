from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from rich.console import Console

from open_endurance_coach.engine.coach import ReviewView
from open_endurance_coach.schemas.decisions import (
    CreateWorkout,
    DecisionReport,
    WorkoutMutation,
)
from open_endurance_coach.store.records import Draft, DraftStatus
from open_endurance_coach.writer.records import ApplyReport

console = Console()


@asynccontextmanager
async def thinking(message: str = "Thinking") -> AsyncIterator[None]:
    if console.is_terminal:
        with console.status(f"[bold cyan]{message}…[/bold cyan]", spinner="dots"):
            yield
    else:
        console.print(f"[dim]{message}…[/dim]")
        yield


def render_report(report: DecisionReport) -> None:
    console.print(f"[bold green]Coach:[/bold green] {report.summary}")
    for finding in report.findings:
        console.print(f"  [dim]- {finding}[/dim]")
    for question in report.questions:
        console.print(f"  [yellow]? {question}[/yellow]")


def render_draft(draft: Draft, *, updated: bool = False, chat: bool = False) -> None:
    render_report(draft.report)
    verb = "updated" if updated else "saved"
    review_hint = f"/review {draft.id}" if chat else f"coach review {draft.id}"
    console.print(f"Draft #{draft.id} {verb} (pending). Review it: {review_hint}")


def render_review(view: ReviewView, *, chat: bool = False) -> None:
    render_report(view.draft.report)
    if view.draft.status is DraftStatus.PENDING:
        for line in view.requested_feedback:
            console.print(f"  [yellow]? {line}[/yellow]")
        if view.requested_feedback:
            hint = f"/feedback {view.draft.id}" if chat else f"coach feedback {view.draft.id}"
            console.print(f'Answer the coach: {hint} "your RPE and notes"')


def mutations_plan_text(mutations: list[WorkoutMutation]) -> str:
    lines = ["Proposed changes:"]
    if not mutations:
        lines.append("  (no calendar changes)")
    for mutation in mutations:
        if isinstance(mutation, CreateWorkout):
            line = f"  - create {mutation.name} on {mutation.start_date_local.isoformat()}"
            if mutation.description:
                line += f": {mutation.description}"
            lines.append(line)
        else:
            lines.append(f"  - {mutation.action} event {mutation.event_id}")
    return "\n".join(lines)


def reject_plan_text(draft_id: int) -> str:
    return (
        f"Draft #{draft_id} - reject: discards the draft and its feedback;"
        " nothing changes on Intervals.icu."
    )


def apply_plan_text(report: ApplyReport) -> str:
    lines: list[str] = []
    for applied in report.decisions:
        lines.append(f"Decision #{applied.decision_id}:")
        for outcome in applied.outcomes:
            target = str(outcome.event_id) if outcome.event_id is not None else outcome.name or ""
            lines.append(f"  - {outcome.action} {outcome.target}: {target}")
    return "\n".join(lines)


def render_apply(report: ApplyReport, *, write: bool) -> None:
    if not report.decisions:
        console.print("No unapplied decisions.")
        return
    if write:
        console.print("[green]Applied:[/green]")
    else:
        console.print("[yellow]DRY RUN - no changes written[/yellow]")
    console.print(apply_plan_text(report))

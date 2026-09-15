from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from rich.console import Console
from rich.markup import escape

from open_endurance_coach.schemas.decisions import (
    CreateRace,
    CreateWorkout,
    DecisionReport,
    DeleteRace,
    Mutation,
    UpdateRace,
    UpdateWorkout,
)
from open_endurance_coach.writer.records import ApplyReport

console = Console()


def print_error(exc: Exception) -> None:
    console.print(f"[red]error:[/red] {escape(str(exc))}")


@asynccontextmanager
async def thinking(message: str = "Thinking") -> AsyncIterator[None]:
    if console.is_terminal:
        with console.status(f"[bold cyan]{message}…[/bold cyan]", spinner="dots"):
            yield
    else:
        console.print(f"[dim]{message}…[/dim]")
        yield


def render_report(report: DecisionReport) -> None:
    console.print(f"[bold green]Coach:[/bold green] {escape(report.summary)}")
    for finding in report.findings:
        console.print(f"  [dim]- {escape(finding)}[/dim]")
    for question in report.questions:
        console.print(f"  [yellow]? {escape(question)}[/yellow]")


def mutations_plan_text(mutations: Sequence[Mutation]) -> str:
    lines = ["Proposed changes:"]
    if not mutations:
        lines.append("  (no calendar changes)")
    for mutation in mutations:
        if isinstance(mutation, CreateRace):
            line = (
                f"  - create {mutation.category} {escape(mutation.name)}"
                f" on {mutation.start_date_local.isoformat()}"
            )
            if mutation.type:
                line += f" ({escape(mutation.type)})"
            if mutation.description:
                line += f": {escape(mutation.description)}"
            lines.append(line)
        elif isinstance(mutation, UpdateRace):
            fields = []
            if mutation.name is not None:
                fields.append(f"name={escape(mutation.name)}")
            if mutation.start_date_local is not None:
                fields.append(f"date={mutation.start_date_local.isoformat()}")
            if mutation.category is not None:
                fields.append(f"category={mutation.category}")
            if mutation.type is not None:
                fields.append(f"type={escape(mutation.type)}")
            if mutation.moving_time is not None:
                fields.append(f"moving_time={mutation.moving_time}")
            if mutation.description is not None:
                fields.append(f"description={escape(mutation.description)}")
            if mutation.icu_training_load is not None:
                fields.append(f"load={mutation.icu_training_load}")
            detail = ", ".join(fields) if fields else "no changes"
            lines.append(f"  - update race event {escape(str(mutation.event_id))}: {detail}")
        elif isinstance(mutation, DeleteRace):
            lines.append(f"  - delete race event {escape(str(mutation.event_id))}")
        elif isinstance(mutation, CreateWorkout):
            line = f"  - create {escape(mutation.name)} on {mutation.start_date_local.isoformat()}"
            if mutation.description:
                line += f": {escape(mutation.description)}"
            lines.append(line)
        elif isinstance(mutation, UpdateWorkout):
            fields = []
            if mutation.name is not None:
                fields.append(f"name={escape(mutation.name)}")
            if mutation.start_date_local is not None:
                fields.append(f"date={mutation.start_date_local.isoformat()}")
            if mutation.moving_time is not None:
                fields.append(f"moving_time={mutation.moving_time}")
            if mutation.description is not None:
                fields.append(f"description={escape(mutation.description)}")
            if mutation.icu_training_load is not None:
                fields.append(f"load={mutation.icu_training_load}")
            detail = ", ".join(fields) if fields else "no changes"
            lines.append(f"  - update event {escape(str(mutation.event_id))}: {detail}")
        else:
            lines.append(f"  - {mutation.action} event {escape(str(mutation.event_id))}")
    return "\n".join(lines)


def apply_plan_text(report: ApplyReport) -> str:
    lines: list[str] = []
    for applied in report.decisions:
        lines.append(f"Decision #{applied.decision_id}:")
        for outcome in applied.outcomes:
            if outcome.event_id is not None:
                lines.append(
                    f"  - {outcome.action} -> {outcome.target}"
                    f" event {escape(str(outcome.event_id))}"
                )
            elif outcome.name:
                lines.append(f"  - {outcome.action} -> {outcome.target}: {escape(outcome.name)}")
            else:
                lines.append(f"  - {outcome.action} -> {outcome.target}")
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

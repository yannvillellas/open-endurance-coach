from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import date

from rich.console import Console
from rich.markup import escape
from rich.theme import Theme

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

THEME = Theme(
    {
        "coach.label": "bold cyan",
        "athlete.label": "bold green",
        "finding": "dim",
        "question": "yellow",
        "plan.frame": "yellow",
        "plan.title": "bold yellow",
        "hint": "dim italic",
        "meta": "dim",
        "warn": "yellow",
        "error": "bold red",
        "success": "green",
    }
)

console = Console(theme=THEME)


def print_error(exc: Exception) -> None:
    console.print(f"[error]error:[/error] {escape(str(exc))}")


@asynccontextmanager
async def thinking(message: str = "Thinking") -> AsyncIterator[None]:
    if console.is_terminal:
        with console.status(f"[meta]{message}…[/meta]", spinner="dots"):
            yield
    else:
        console.print(f"[meta]{message}…[/meta]")
        yield


def render_report(report: DecisionReport) -> None:
    console.print(f"[coach.label]Coach:[/coach.label] {escape(report.summary)}")
    for finding in report.findings:
        console.print(f"  [finding]- {escape(finding)}[/finding]")
    for question in report.questions:
        console.print(f"  [question]? {escape(question)}[/question]")


def _mutation_date(mutation: Mutation) -> date | None:
    return getattr(mutation, "start_date_local", None)


def _inline_description(description: str | None) -> str:
    if description and "\n" not in description:
        return f": {escape(description)}"
    return ""


def _nested_description(description: str | None) -> list[str]:
    if not description or "\n" not in description:
        return []
    return [
        f"      {escape(line.rstrip())}" if line.strip() else ""
        for line in description.splitlines()
    ]


def _mutation_lines(mutation: Mutation) -> list[str]:
    if isinstance(mutation, CreateRace):
        line = f"    - create {mutation.category} {escape(mutation.name)}"
        details = []
        if mutation.type:
            details.append(escape(mutation.type))
        if mutation.moving_time is not None:
            details.append(f"moving_time={mutation.moving_time}")
        if mutation.distance is not None:
            details.append(f"distance={mutation.distance:g}m")
        if mutation.icu_training_load is not None:
            details.append(f"load={mutation.icu_training_load:g}")
        if details:
            line += f" ({', '.join(details)})"
        line += _inline_description(mutation.description)
        return [line, *_nested_description(mutation.description)]
    if isinstance(mutation, UpdateRace):
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
        if mutation.distance is not None:
            fields.append(f"distance={mutation.distance:g}m")
        if mutation.description is not None and "\n" not in mutation.description:
            fields.append(f"description={escape(mutation.description)}")
        if mutation.icu_training_load is not None:
            fields.append(f"load={mutation.icu_training_load:g}")
        detail = ", ".join(fields) if fields else "no changes"
        line = f"    - update race event {escape(str(mutation.event_id))}: {detail}"
        return [line, *_nested_description(mutation.description)]
    if isinstance(mutation, DeleteRace):
        return [f"    - delete race event {escape(str(mutation.event_id))}"]
    if isinstance(mutation, CreateWorkout):
        line = f"    - create {escape(mutation.name)}"
        details = []
        if mutation.type:
            details.append(escape(mutation.type))
        if mutation.moving_time is not None:
            details.append(f"moving_time={mutation.moving_time}")
        if mutation.icu_training_load is not None:
            details.append(f"load={mutation.icu_training_load:g}")
        if details:
            line += f" ({', '.join(details)})"
        line += _inline_description(mutation.description)
        return [line, *_nested_description(mutation.description)]
    if isinstance(mutation, UpdateWorkout):
        fields = []
        if mutation.name is not None:
            fields.append(f"name={escape(mutation.name)}")
        if mutation.start_date_local is not None:
            fields.append(f"date={mutation.start_date_local.isoformat()}")
        if mutation.moving_time is not None:
            fields.append(f"moving_time={mutation.moving_time}")
        if mutation.description is not None and "\n" not in mutation.description:
            fields.append(f"description={escape(mutation.description)}")
        if mutation.icu_training_load is not None:
            fields.append(f"load={mutation.icu_training_load:g}")
        detail = ", ".join(fields) if fields else "no changes"
        line = f"    - update event {escape(str(mutation.event_id))}: {detail}"
        return [line, *_nested_description(mutation.description)]
    return [f"    - {mutation.action} event {escape(str(mutation.event_id))}"]


def mutations_plan_text(mutations: Sequence[Mutation]) -> str:
    lines = ["Proposed changes:"]
    if not mutations:
        lines.append("  (no calendar changes)")
        return "\n".join(lines)
    grouped: dict[date, list[Mutation]] = {}
    undated: list[Mutation] = []
    for mutation in mutations:
        day = _mutation_date(mutation)
        if day is None:
            undated.append(mutation)
        else:
            grouped.setdefault(day, []).append(mutation)
    for day in sorted(grouped):
        lines.append(f"  {day.isoformat()}")
        for mutation in grouped[day]:
            lines.extend(_mutation_lines(mutation))
    if undated:
        lines.append("  (no date)")
        for mutation in undated:
            lines.extend(_mutation_lines(mutation))
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


def render_apply(report: ApplyReport) -> None:
    if not report.decisions:
        console.print("No unapplied decisions.")
        return
    console.print("[success]Applied:[/success]")
    console.print(apply_plan_text(report))
    skipped = [
        (decision.decision_id, len(decision.skipped), sorted(set(decision.skipped)))
        for decision in report.decisions
        if decision.skipped
    ]
    for decision_id, count, reasons in skipped:
        console.print(
            f"[warn]Decision #{decision_id}: {count} mutation(s) skipped"
            f" ({', '.join(reasons)}) — the calendar was only partially updated.[/warn]"
        )

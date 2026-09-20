import re
from collections.abc import AsyncIterator, Mapping, Sequence
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

console = Console(theme=THEME, highlight=False)


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


_TOPIC_RE = re.compile(r"^([^:]{1,32}?):\s")


def split_finding_topic(finding: str) -> tuple[str, str]:
    match = _TOPIC_RE.match(finding)
    if match is None:
        return "", finding
    topic = match.group(1).strip()
    if not topic or len(topic.split()) > 3:
        return "", finding
    return topic, finding[match.end(1) :]


def _print_finding(finding: str) -> None:
    topic, _ = split_finding_topic(finding)
    lines = wrap_plan_text(f"  - {finding}", console.width).splitlines()
    if not lines:
        return
    body = lines[0][4:]
    if topic and body.startswith(topic):
        head = f"  [meta]-[/meta] [coach.label]{escape(topic)}[/coach.label]"
        console.print(head + escape(body[len(topic) :]))
    else:
        console.print(f"  [meta]-[/meta] {escape(body)}")
    for extra in lines[1:]:
        console.print(escape(extra))


def _print_question(question: str) -> None:
    lines = wrap_plan_text(f"  ? {question}", console.width).splitlines()
    for index, line in enumerate(lines):
        if index == 0:
            console.print(f"  [question]? {escape(line[4:])}[/question]")
        else:
            console.print(f"[question]{escape(line)}[/question]")


def render_report(report: DecisionReport) -> None:
    findings = [finding for finding in report.findings if finding.strip()]
    questions = [question for question in report.questions if question.strip()]
    console.print()
    console.print("[coach.label]Coach:[/coach.label]")
    for line in wrap_plan_text(f"  {report.summary}", console.width, hanging=2).splitlines():
        console.print(escape(line))
    if findings:
        console.print()
        console.print("[coach.label]Evidence:[/coach.label]")
        for finding in findings:
            _print_finding(finding)
    if questions:
        console.print()
        console.print("[coach.label]Open questions:[/coach.label]")
        for question in questions:
            _print_question(question)


def wrap_plan_text(text: str, width: int, *, hanging: int | None = None) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if not line.strip() or len(line) <= width:
            lines.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        requested = hanging if hanging is not None else indent + 2
        hanging_text = " " * max(requested, indent)
        words = line.split()
        current = " " * indent + words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= width:
                current += " " + word
            else:
                lines.append(current)
                current = hanging_text + word
        lines.append(current)
    return "\n".join(lines)


def _mutation_date(
    mutation: Mutation, event_dates: Mapping[str, date] | None = None
) -> date | None:
    day = getattr(mutation, "start_date_local", None)
    if day is not None:
        return day
    event_id = getattr(mutation, "event_id", None)
    if event_id is not None and event_dates is not None:
        return event_dates.get(str(event_id))
    return None


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
        if mutation.distance is not None:
            details.append(f"distance={mutation.distance:g}m")
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
        if mutation.distance is not None:
            fields.append(f"distance={mutation.distance:g}m")
        if mutation.description is not None and "\n" not in mutation.description:
            fields.append(f"description={escape(mutation.description)}")
        if mutation.icu_training_load is not None:
            fields.append(f"load={mutation.icu_training_load:g}")
        detail = ", ".join(fields) if fields else "no changes"
        line = f"    - update event {escape(str(mutation.event_id))}: {detail}"
        return [line, *_nested_description(mutation.description)]
    return [f"    - {mutation.action} event {escape(str(mutation.event_id))}"]


def mutations_plan_text(
    mutations: Sequence[Mutation], *, event_dates: Mapping[str, date] | None = None
) -> str:
    lines = ["Proposed changes:"]
    if not mutations:
        lines.append("  (no calendar changes)")
        return "\n".join(lines)
    grouped: dict[date, list[Mutation]] = {}
    undated: list[Mutation] = []
    for mutation in mutations:
        day = _mutation_date(mutation, event_dates)
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
            for note in outcome.drift:
                lines.append(f"    note: {escape(note)}")
    return "\n".join(lines)


def render_apply(report: ApplyReport) -> None:
    if not report.decisions:
        console.print("No unapplied decisions.")
        return
    console.print()
    console.print("[success]Applied:[/success]")
    console.print(apply_plan_text(report))

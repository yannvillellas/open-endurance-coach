import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import typer
from pydantic import TypeAdapter, ValidationError
from rich.console import Console

from open_endurance_coach.clients.intervals import IntervalsClient
from open_endurance_coach.clients.llm import LlmClient, LlmError
from open_endurance_coach.clients.providers import build_registry
from open_endurance_coach.config import get_settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.schemas.decisions import DecisionReport, WorkoutMutation
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import Draft

app = typer.Typer(no_args_is_help=True)
console = Console()
_mutations_adapter = TypeAdapter(list[WorkoutMutation])

DEFAULT_ANALYZE_FOCUS = "Analyze my recent training"


async def _with_engine(callback: Callable[[CoachEngine], Awaitable[None]]) -> None:
    settings = get_settings()
    intervals = IntervalsClient(settings)
    providers = build_registry(settings)
    llm = LlmClient(settings, providers)
    store = CoachStore(settings.database_path)
    engine = CoachEngine(settings, store, intervals, llm)
    try:
        await callback(engine)
    finally:
        await intervals.aclose()
        for provider in providers.values():
            await provider.aclose()
        store.close()


def _run(callback: Callable[[CoachEngine], Awaitable[None]]) -> None:
    try:
        asyncio.run(_with_engine(callback))
    except (LlmError, ValueError, RuntimeError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _render_report(report: DecisionReport) -> None:
    console.print(f"[bold green]Coach:[/bold green] {report.summary}")
    for finding in report.findings:
        console.print(f"  [dim]- {finding}[/dim]")
    for question in report.questions:
        console.print(f"  [yellow]? {question}[/yellow]")


def _render_draft(draft: Draft) -> None:
    _render_report(draft.report)
    console.print(f"Draft #{draft.id} saved (pending). Review it: coach review {draft.id}")


@app.command()
def ask(
    focus: str = typer.Argument(..., help="Your question or analysis focus"),
    feedback: str | None = typer.Option(None, "--feedback", help="Subjective context to inject"),
) -> None:
    async def run(engine: CoachEngine) -> None:
        _render_draft(await engine.analyze(focus, user_feedback=feedback))

    _run(run)


@app.command()
def analyze(
    focus: str = typer.Argument(DEFAULT_ANALYZE_FOCUS),
    feedback: str | None = typer.Option(None, "--feedback", help="Subjective context to inject"),
) -> None:
    async def run(engine: CoachEngine) -> None:
        _render_draft(await engine.analyze(focus, user_feedback=feedback))

    _run(run)


@app.command()
def review(
    draft_id: int | None = typer.Argument(None, help="Draft id; omit to list pending"),
) -> None:
    async def run(engine: CoachEngine) -> None:
        if draft_id is None:
            drafts = engine.pending_drafts()
            if not drafts:
                console.print("No pending drafts.")
                return
            for draft in drafts:
                console.print(
                    f"Draft #{draft.id} ([cyan]{draft.status.value}[/cyan]): {draft.report.summary}"
                )
            return
        view = engine.review(draft_id)
        _render_report(view.draft.report)
        for line in view.requested_feedback:
            console.print(f"  [yellow]? {line}[/yellow]")

    _run(run)


@app.command()
def approve(
    draft_id: int = typer.Argument(...),
    mutations: str | None = typer.Option(
        None,
        "--mutations",
        help="JSON array of workout mutations replacing the coach's proposals",
    ),
) -> None:
    override = _parse_mutations(mutations) if mutations is not None else None

    async def run(engine: CoachEngine) -> None:
        decision = engine.approve(draft_id, mutations=override)
        console.print(
            f"Decision #{decision.id} recorded from draft #{draft_id}"
            f" ({len(decision.report.mutations)} mutations)."
        )

    _run(run)


@app.command()
def reject(draft_id: int = typer.Argument(...)) -> None:
    async def run(engine: CoachEngine) -> None:
        engine.reject(draft_id)
        console.print(f"Draft #{draft_id} rejected.")

    _run(run)


def _parse_mutations(raw: str) -> list[WorkoutMutation]:
    try:
        payload: Any = json.loads(raw)
        return _mutations_adapter.validate_python(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise typer.BadParameter(f"invalid mutations JSON: {exc}") from exc


def main() -> None:
    app()

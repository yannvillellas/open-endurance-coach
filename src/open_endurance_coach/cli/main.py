import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import typer
from pydantic import TypeAdapter, ValidationError

# imported before app.add_typer below; cli.chat reaches cli.main lazily to avoid a cycle
from open_endurance_coach.cli.chat import chat_app
from open_endurance_coach.cli.rendering import console, render_apply, render_draft, render_review
from open_endurance_coach.clients.intervals import IntervalsClient
from open_endurance_coach.clients.llm import LlmClient, LlmError
from open_endurance_coach.clients.providers import build_registry
from open_endurance_coach.config import get_settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.schemas.decisions import WorkoutMutation
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.writer.calendar import CalendarWriter

app = typer.Typer(no_args_is_help=True)
app.add_typer(chat_app)
_mutations_adapter = TypeAdapter(list[WorkoutMutation])

DEFAULT_ANALYZE_FOCUS = "Analyze my recent training"


async def _with_engine(callback: Callable[[CoachEngine], Awaitable[None]]) -> None:
    try:
        settings = get_settings()
    except ValidationError as exc:
        console.print(
            "[red]error:[/red] configuration missing: is there a readable .env file"
            " in the current directory with all required keys?"
        )
        console.print(f"[dim]{exc}[/dim]")
        raise typer.Exit(code=1) from None
    intervals = IntervalsClient(settings)
    providers = build_registry(settings)
    llm = LlmClient(settings, providers)
    store = CoachStore(settings.database_path)
    writer = CalendarWriter(intervals)
    engine = CoachEngine(settings, store, intervals, llm, writer=writer)
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


@app.command()
def ask(
    focus: str = typer.Argument(..., help="Your question or analysis focus"),
    feedback: str | None = typer.Option(None, "--feedback", help="Subjective context to inject"),
) -> None:
    async def run(engine: CoachEngine) -> None:
        render_draft(await engine.analyze(focus, user_feedback=feedback))

    _run(run)


@app.command()
def analyze(
    focus: str = typer.Argument(DEFAULT_ANALYZE_FOCUS),
    feedback: str | None = typer.Option(None, "--feedback", help="Subjective context to inject"),
) -> None:
    async def run(engine: CoachEngine) -> None:
        render_draft(await engine.analyze(focus, user_feedback=feedback))

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
        render_review(engine.review(draft_id))

    _run(run)


@app.command()
def feedback(
    draft_id: int = typer.Argument(...),
    text: str = typer.Argument(...),
) -> None:
    async def run(engine: CoachEngine) -> None:
        render_draft(await engine.submit_feedback(draft_id, text), updated=True)

    _run(run)


@app.command()
def approve(
    draft_id: int = typer.Argument(...),
    mutations_file: str | None = typer.Option(
        None,
        "--mutations-file",
        help="JSON file with workout mutations replacing the coach's proposals",
    ),
) -> None:
    override = _read_mutations(mutations_file) if mutations_file is not None else None

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


@app.command()
def apply(
    decision_id: int | None = typer.Argument(None, help="Decision id; omit to apply all unapplied"),
    write: bool = typer.Option(
        False, "--write", help="Write to the calendar (default is a dry-run)"
    ),
) -> None:
    async def run(engine: CoachEngine) -> None:
        render_apply(await engine.apply(decision_id, dry_run=not write), write=write)

    _run(run)


def _read_mutations(path: str) -> list[WorkoutMutation]:
    try:
        with open(path, encoding="utf-8") as handle:
            payload: Any = json.load(handle)
        return _mutations_adapter.validate_python(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise typer.BadParameter(f"invalid mutations file: {exc}") from exc


def main() -> None:
    app()

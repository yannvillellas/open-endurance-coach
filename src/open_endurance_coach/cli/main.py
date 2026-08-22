import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable
from typing import Any

import typer
from pydantic import TypeAdapter, ValidationError

# imported before app.add_typer below; cli.chat reaches cli.main lazily to avoid a cycle
from open_endurance_coach.chat.gate import PlanSnapshot
from open_endurance_coach.cli.chat import chat_app
from open_endurance_coach.cli.confirmation import run_confirmation
from open_endurance_coach.cli.rendering import (
    apply_plan_text,
    console,
    escape,
    mutations_plan_text,
    reject_plan_text,
    render_apply,
    render_draft,
    render_review,
    thinking,
)
from open_endurance_coach.clients.intervals import IntervalsClient
from open_endurance_coach.clients.llm import LlmClient, LlmError
from open_endurance_coach.clients.providers import build_registry
from open_endurance_coach.config import get_settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.schemas.decisions import WorkoutMutation
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.store.records import Draft, DraftStatus
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
    except (LlmError, ValueError, RuntimeError, sqlite3.Error) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _approve_snapshot(
    engine: CoachEngine, draft_id: int, override: list[WorkoutMutation] | None
) -> PlanSnapshot:
    view = engine.review(draft_id)
    if view.draft.status is not DraftStatus.PENDING:
        raise ValueError(
            f"draft {draft_id} is {view.draft.status.value}; only pending drafts can be approved"
        )
    mutations = override if override is not None else view.draft.report.mutations
    return PlanSnapshot(
        action="approve",
        plan_text=mutations_plan_text(mutations),
        draft_id=draft_id,
    )


def _reject_snapshot(engine: CoachEngine, draft_id: int) -> PlanSnapshot:
    view = engine.review(draft_id)
    if view.draft.status is not DraftStatus.PENDING:
        raise ValueError(
            f"draft {draft_id} is {view.draft.status.value}; only pending drafts can be rejected"
        )
    return PlanSnapshot(action="reject", plan_text=reject_plan_text(draft_id), draft_id=draft_id)


async def _apply_snapshot(engine: CoachEngine, decision_id: int | None) -> PlanSnapshot | None:
    report = await engine.apply(decision_id, dry_run=True)
    if not report.decisions:
        console.print("No unapplied decisions.")
        return None
    return PlanSnapshot(
        action="apply",
        plan_text=f"Decision(s) - write to calendar:\n{apply_plan_text(report)}",
        draft_id=None,
        decision_id=decision_id,
        write=True,
    )


async def _execute_approve(
    engine: CoachEngine, draft_id: int, override: list[WorkoutMutation] | None
) -> None:
    decision = engine.approve(draft_id, mutations=override)
    console.print(
        f"Decision #{decision.id} recorded from draft #{draft_id}"
        f" ({len(decision.report.mutations)} mutations)."
    )


async def _execute_reject(engine: CoachEngine, draft_id: int) -> None:
    engine.reject(draft_id)
    console.print(f"Draft #{draft_id} rejected.")


async def _execute_apply(engine: CoachEngine, decision_id: int | None, write: bool) -> None:
    render_apply(await engine.apply(decision_id, dry_run=not write), write=write)


def _approve_restate(override: list[WorkoutMutation] | None) -> Callable[[Draft], str]:
    def restate(draft: Draft) -> str:
        mutations = override if override is not None else draft.report.mutations
        return mutations_plan_text(mutations)

    return restate


@app.command()
def ask(
    focus: str = typer.Argument(..., help="Your question or analysis focus"),
    feedback: str | None = typer.Option(None, "--feedback", help="Subjective context to inject"),
) -> None:
    async def run(engine: CoachEngine) -> None:
        async with thinking():
            render_draft(await engine.analyze(focus, user_feedback=feedback))

    _run(run)


@app.command()
def analyze(
    focus: str = typer.Argument(DEFAULT_ANALYZE_FOCUS),
    feedback: str | None = typer.Option(None, "--feedback", help="Subjective context to inject"),
) -> None:
    async def run(engine: CoachEngine) -> None:
        async with thinking():
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
                    f"Draft #{draft.id} ([cyan]{draft.status.value}[/cyan]):"
                    f" {escape(draft.report.summary)}"
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
        async with thinking():
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
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt"),
) -> None:
    override = _read_mutations(mutations_file) if mutations_file is not None else None

    async def run(engine: CoachEngine) -> None:
        if yes:
            await _execute_approve(engine, draft_id, override)
            return
        snapshot = _approve_snapshot(engine, draft_id, override)

        async def execute(current: CoachEngine) -> None:
            await _execute_approve(current, draft_id, override)

        await run_confirmation(
            engine, snapshot, executor=execute, restate=_approve_restate(override)
        )

    _run(run)


@app.command()
def reject(
    draft_id: int = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt"),
) -> None:
    async def run(engine: CoachEngine) -> None:
        if yes:
            await _execute_reject(engine, draft_id)
            return
        snapshot = _reject_snapshot(engine, draft_id)
        await run_confirmation(
            engine,
            snapshot,
            executor=lambda current: _execute_reject(current, draft_id),
            restate=lambda draft: reject_plan_text(draft.id),
        )

    _run(run)


@app.command()
def apply(
    decision_id: int | None = typer.Argument(None, help="Decision id; omit to apply all unapplied"),
    write: bool = typer.Option(
        False, "--write", help="Write to the calendar (default is a dry-run)"
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt"),
) -> None:
    async def run(engine: CoachEngine) -> None:
        if not write or yes:
            await _execute_apply(engine, decision_id, write)
            return
        snapshot = await _apply_snapshot(engine, decision_id)
        if snapshot is None:
            return
        await run_confirmation(
            engine,
            snapshot,
            executor=lambda current: _execute_apply(current, decision_id, write),
        )

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

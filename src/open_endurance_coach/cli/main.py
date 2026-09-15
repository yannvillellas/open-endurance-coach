import asyncio
from collections.abc import Awaitable, Callable

import typer
from pydantic import ValidationError

from open_endurance_coach.chat.gate import RECOVERABLE_EXCEPTIONS
from open_endurance_coach.cli.rendering import console, print_error
from open_endurance_coach.clients.intervals import IntervalsClient
from open_endurance_coach.clients.llm import LlmClient
from open_endurance_coach.clients.providers import build_registry
from open_endurance_coach.config import get_settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.store.db import CoachStore
from open_endurance_coach.writer.calendar import CalendarWriter

app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def _default_entry(
    ctx: typer.Context,
    provider: str | None = typer.Option(
        None, "--provider", "-p", help="LLM provider (ovh | deepseek)"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="LLM model override"),
) -> None:
    if ctx.invoked_subcommand is None:
        from open_endurance_coach.cli.chat import start_chat

        start_chat(provider=provider, model=model)


async def _with_engine(
    callback: Callable[[CoachEngine], Awaitable[None]],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    try:
        settings = get_settings()
    except ValidationError as exc:
        console.print(
            "[red]error:[/red] configuration missing: is there a readable .env file"
            " in the current directory with all required keys?"
        )
        console.print(f"[dim]{exc}[/dim]")
        raise typer.Exit(code=1) from None
    settings = settings.with_llm_override(provider=provider, model=model)
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
        for llm_provider in providers.values():
            await llm_provider.aclose()
        store.close()


def _run(
    callback: Callable[[CoachEngine], Awaitable[None]],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    try:
        asyncio.run(_with_engine(callback, provider=provider, model=model))
    except RECOVERABLE_EXCEPTIONS as exc:
        print_error(exc)
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()

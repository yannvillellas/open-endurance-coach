from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from rich.prompt import Prompt

from open_endurance_coach.chat.gate import (
    Cancelled,
    Declined,
    Discuss,
    Feedback,
    Ignored,
    PlanSnapshot,
    Proceed,
    handle,
)
from open_endurance_coach.cli.rendering import (
    console,
    mutations_plan_text,
    render_draft,
    render_report,
    thinking,
)
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.store.records import Draft

Executor = Callable[[CoachEngine], Awaitable[None]]


@dataclass(frozen=True)
class Done:
    pass


def restate_mutations(draft: Draft) -> str:
    return mutations_plan_text(draft.report.mutations)


def prompt_plan(snapshot: PlanSnapshot) -> None:
    console.print("[bold yellow]Confirm? Reply with exactly yes or no.[/bold yellow]")
    console.print(snapshot.plan_text)
    console.print("[dim](yes / no / cancel)[/dim]")


async def respond(
    engine: CoachEngine,
    snapshot: PlanSnapshot,
    line: str,
    *,
    executor: Executor,
    chat: bool,
    discuss_message: str | None = None,
    restate: Callable[[Draft], str] = restate_mutations,
    on_discuss: Callable[[str], Awaitable[None]] | None = None,
    on_feedback: Callable[[str, Draft], Awaitable[bool | None]] | None = None,
) -> Done | PlanSnapshot:
    match handle(line, snapshot):
        case Proceed():
            await executor(engine)
            return Done()
        case Declined():
            console.print("[yellow]Nothing changed.[/yellow]")
            return Done()
        case Cancelled():
            console.print("[yellow]Cancelled. Nothing changed.[/yellow]")
            return Done()
        case Ignored():
            return snapshot
        case Feedback(feedback):
            assert snapshot.draft_id is not None
            async with thinking():
                updated = await engine.submit_feedback(snapshot.draft_id, feedback)
            if chat:
                render_report(updated.report)
            else:
                render_draft(updated, updated=True, chat=False)
            if on_feedback is not None and await on_feedback(feedback, updated):
                return Done()
            return replace(snapshot, plan_text=restate(updated))
        case Discuss(line):
            if on_discuss is not None:
                await on_discuss(line)
            elif discuss_message is not None:
                console.print(discuss_message)
            return snapshot


async def run_confirmation(
    engine: CoachEngine,
    snapshot: PlanSnapshot,
    *,
    executor: Executor,
    restate: Callable[[Draft], str] = restate_mutations,
) -> None:
    current = snapshot
    while True:
        prompt_plan(current)
        try:
            line = Prompt.ask("[bold cyan]you[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("[yellow]Cancelled. Nothing changed.[/yellow]")
            return
        words = line.strip().casefold().split()
        if words and words[0] in {"/exit", "/quit"}:
            console.print("[yellow]Cancelled. Nothing changed.[/yellow]")
            return
        step = await respond(
            engine,
            current,
            line,
            executor=executor,
            chat=False,
            discuss_message=(
                "[yellow]Discussion lives in `coach chat`; reply yes/no/cancel here.[/yellow]"
            ),
            restate=restate,
        )
        if isinstance(step, Done):
            return
        current = step

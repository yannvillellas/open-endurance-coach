from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from rich.panel import Panel

from open_endurance_coach.chat.gate import (
    Declined,
    Feedback,
    Ignored,
    InvalidSelection,
    PlanItem,
    PlanSnapshot,
    Proceed,
    ProceedSubset,
    handle,
)
from open_endurance_coach.cli.rendering import (
    console,
    escape,
    render_report,
    thinking,
    wrap_plan_text,
)
from open_endurance_coach.clients.llm import LlmMessage
from open_endurance_coach.engine.coach import CoachEngine, FeedbackOutcome
from open_endurance_coach.store.records import Draft

Executor = Callable[[CoachEngine, tuple[int, ...] | None], Awaitable[None]]


@dataclass(frozen=True)
class Done:
    pass


def prompt_plan(snapshot: PlanSnapshot) -> None:
    console.print()
    console.print(
        "[plan.title]Confirm? Reply exactly yes to apply, no to discard.\n"
        "Or describe a change to revise the plan.[/plan.title]"
    )
    console.print(
        "[hint]Approve only some items with yes 1 3, yes except 2, or yes except thursday.[/hint]"
    )
    console.print(
        Panel(
            console.render_str(wrap_plan_text(snapshot.plan_text, max(20, console.width - 4))),
            title="Proposal",
            title_align="left",
            border_style="plan.frame",
            padding=(0, 1),
        )
    )
    console.print("[hint](yes / no, or describe a change)[/hint]")


async def respond(
    engine: CoachEngine,
    snapshot: PlanSnapshot,
    line: str,
    *,
    executor: Executor,
    restate: Callable[[Draft], Awaitable[tuple[str, tuple[PlanItem, ...]]]],
    on_feedback: Callable[[str, FeedbackOutcome], Awaitable[bool | None]] | None = None,
    assume_answers: bool = False,
    history: list[LlmMessage] | None = None,
) -> Done | PlanSnapshot:
    match handle(line, snapshot):
        case Proceed():
            await executor(engine, None)
            return Done()
        case ProceedSubset(indices):
            await executor(engine, indices)
            return Done()
        case InvalidSelection(reason):
            console.print(f"[warn]{escape(reason)} Nothing changed.[/warn]")
            return snapshot
        case Declined():
            engine.reject_draft(snapshot.draft_id)
            console.print("[warn]Nothing changed.[/warn]")
            return Done()
        case Ignored():
            return snapshot
        case Feedback(feedback):
            async with thinking():
                outcome = await engine.submit_feedback(
                    snapshot.draft_id,
                    feedback,
                    focus=feedback,
                    assume=assume_answers,
                    history=history,
                )
            render_report(outcome.report)
            if on_feedback is not None and await on_feedback(feedback, outcome):
                return Done()
            plan_text, items = await restate(outcome.draft)
            return replace(snapshot, plan_text=plan_text, items=items)

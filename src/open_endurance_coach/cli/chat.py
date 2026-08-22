import re

import typer
from rich.prompt import Prompt

from open_endurance_coach.chat.dispatch import (
    Command,
    Confirmation,
    Converse,
    Exit,
    Ignore,
    UnknownCommand,
    dispatch,
)
from open_endurance_coach.chat.gate import PlanSnapshot
from open_endurance_coach.chat.history import ChatSession, assistant_turn
from open_endurance_coach.chat.state import ChatState
from open_endurance_coach.cli.confirmation import Done, prompt_plan, respond
from open_endurance_coach.cli.rendering import (
    console,
    mutations_plan_text,
    render_apply,
    render_report,
)
from open_endurance_coach.clients.llm import LlmError
from open_endurance_coach.config import Settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.extractors.deep import detect_deep_query
from open_endurance_coach.schemas.decisions import WorkoutMutation
from open_endurance_coach.store.records import Draft

chat_app = typer.Typer()

HELP_TEXT = (
    "Just talk to the coach. When he proposes calendar changes, answer with\n"
    "exactly yes or no (anything else is a change request and nothing is written).\n"
    "/analyze [focus]       force a fresh analysis\n"
    "/clear                 forget this session's memory\n"
    "/help                  show this help\n"
    "/exit, /quit           leave the chat\n"
)

_ANALYZE_RE = re.compile(r"\b(analy[sz]e|review|assess|check|plan)\b", re.IGNORECASE)


def _analysis_due(session: ChatSession, text: str) -> bool:
    return (
        session.context is None
        or detect_deep_query(text) is not None
        or _ANALYZE_RE.search(text) is not None
    )


def _open_proposal(draft_id: int, mutations: list[WorkoutMutation]) -> ChatState:
    snapshot = PlanSnapshot(
        action="approve",
        plan_text="Apply this to Intervals.icu:\n" + mutations_plan_text(draft_id, mutations),
        draft_id=draft_id,
    )
    return _enter_confirmation(snapshot)


def _enter_confirmation(snapshot: PlanSnapshot) -> ChatState:
    prompt_plan(snapshot)
    return ChatState(plan=snapshot)


async def _analyze_line(engine: CoachEngine, session: ChatSession, focus: str) -> ChatState | None:
    draft = await engine.analyze(focus)
    render_report(draft.report)
    session.context = draft.context
    session.append(focus, assistant_turn(draft.report).content)
    if draft.report.mutations:
        return _open_proposal(draft.id, draft.report.mutations)
    console.print("[dim]Answer my questions here if you like.[/dim]")
    return None


async def _handle_converse(
    engine: CoachEngine, session: ChatSession, text: str
) -> ChatState | None:
    try:
        if _analysis_due(session, text):
            return await _analyze_line(engine, session, text)
        reply = await engine.converse(text, history=session.history, context=session.context)
        console.print("[bold green]Coach:[/bold green]", end=" ")
        console.print(reply, markup=False)
        session.append(text, reply)
        return None
    except (LlmError, ValueError, RuntimeError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        return None


async def _apply_proposal(engine: CoachEngine, draft_id: int) -> None:
    decision = engine.approve(draft_id)
    render_apply(await engine.apply(decision.id), write=True)


async def _handle_proposal(
    engine: CoachEngine, state: ChatState, line: str, session: ChatSession
) -> ChatState:
    assert state.plan is not None
    snapshot = state.plan
    draft_id = snapshot.draft_id
    assert draft_id is not None

    async def execute(current: CoachEngine) -> None:
        await _apply_proposal(current, draft_id)

    async def feedback(line: str, updated: Draft) -> bool | None:
        session.append(line, assistant_turn(updated.report).content)
        if not updated.report.mutations:
            console.print("[yellow]No changes proposed anymore.[/yellow]")
            return True
        return None

    def restate(draft: Draft) -> str:
        return "Apply this to Intervals.icu:\n" + mutations_plan_text(
            draft.id, draft.report.mutations
        )

    try:
        step = await respond(
            engine,
            snapshot,
            line,
            executor=execute,
            chat=True,
            on_feedback=feedback,
            restate=restate,
        )
    except (LlmError, ValueError, RuntimeError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        return ChatState()
    if isinstance(step, Done):
        return ChatState()
    return _enter_confirmation(step)


async def _run_command(
    engine: CoachEngine, name: str, args: list[str], session: ChatSession
) -> ChatState | None:
    if name == "help":
        console.print(HELP_TEXT, markup=False)
        return None
    if name == "clear":
        session.history = []
        session.context = None
        console.print("Memory cleared.")
        return None
    if name == "analyze":
        from open_endurance_coach.cli import main as cli_main

        focus = " ".join(args) if args else cli_main.DEFAULT_ANALYZE_FOCUS
        try:
            return await _analyze_line(engine, session, focus)
        except (LlmError, ValueError, RuntimeError) as exc:
            console.print(f"[red]error:[/red] {exc}")
    return None


async def run_chat(engine: CoachEngine, settings: Settings, *, fresh: bool = False) -> None:
    session = ChatSession(cap=settings.chat_history_max_tokens)
    if not fresh:
        session.seed(
            engine.recent_history(settings.chat_history_turns),
            max_tokens=settings.chat_history_max_tokens,
        )
    state = ChatState()
    remembered = sum(1 for turn in session.history if turn.role == "user")
    console.print("Chat with the coach. /help lists commands.")
    if remembered:
        console.print(f"[dim]Remembering {remembered} past exchanges.[/dim]")
    while True:
        try:
            line = Prompt.ask("[bold cyan]you[/bold cyan]")
        except EOFError:
            if state.plan is not None:
                console.print("[yellow]Cancelled. Nothing changed.[/yellow]")
            console.print("bye")
            return
        except KeyboardInterrupt:
            if state.plan is not None:
                console.print("[yellow]Cancelled. Nothing changed.[/yellow]")
                state = ChatState()
                continue
            console.print("bye")
            return
        match dispatch(line, state):
            case Ignore():
                continue
            case Exit():
                console.print("bye")
                return
            case Converse(text=text):
                state = await _handle_converse(engine, session, text) or state
            case UnknownCommand():
                console.print("[red]Unknown command.[/red]")
                console.print(HELP_TEXT, markup=False)
            case Confirmation(line=line):
                state = await _handle_proposal(engine, state, line, session)
            case Command(name=name, args=args):
                state = await _run_command(engine, name, args, session) or state


@chat_app.command()
def chat(
    fresh: bool = typer.Option(False, "--fresh", help="Start without seeded memory"),
) -> None:
    from open_endurance_coach.cli import main as cli_main

    async def run(engine: CoachEngine) -> None:
        await run_chat(engine, cli_main.get_settings(), fresh=fresh)

    cli_main._run(run)

import typer
from rich.console import Console
from rich.prompt import Prompt

from open_endurance_coach.chat.dispatch import (
    Command,
    Converse,
    Exit,
    Ignore,
    UnknownCommand,
    dispatch,
)
from open_endurance_coach.chat.state import ChatState
from open_endurance_coach.clients.llm import LlmError
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.store.records import DraftStatus

chat_app = typer.Typer()
console = Console()

HELP_TEXT = (
    "/analyze [focus]       run a full analysis and save a draft\n"
    "/review [id]           list pending drafts, or inspect one\n"
    "/feedback <id> <text>  answer the coach's questions on a draft\n"
    "/approve <id>          approve a draft (confirmation-gated, wired in 2b)\n"
    "/reject <id>           reject a draft (confirmation-gated, wired in 2b)\n"
    "/apply [id] [--write]  apply decisions (confirmation-gated, wired in 2b)\n"
    "/help                  show this help\n"
    "/exit, /quit           leave the chat\n"
    "Any other line is a conversation turn (wired in 4c)."
)


def _parse_id(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _render_pending(engine: CoachEngine) -> None:
    drafts = engine.pending_drafts()
    if not drafts:
        console.print("No pending drafts.")
        return
    for draft in drafts:
        console.print(
            f"Draft #{draft.id} ([cyan]{draft.status.value}[/cyan]): {draft.report.summary}"
        )


def _render_review(engine: CoachEngine, draft_id: int) -> None:
    from open_endurance_coach.cli import main as cli_main

    view = engine.review(draft_id)
    cli_main._render_report(view.draft.report)
    if view.draft.status is DraftStatus.PENDING:
        for line in view.requested_feedback:
            console.print(f"  [yellow]? {line}[/yellow]")
        if view.requested_feedback:
            console.print(f'Answer the coach: /feedback {draft_id} "your RPE and notes"')


async def _run_command(engine: CoachEngine, name: str, args: list[str]) -> None:
    if name == "help":
        console.print(HELP_TEXT)
        return
    try:
        if name == "analyze":
            from open_endurance_coach.cli import main as cli_main

            focus = " ".join(args) if args else cli_main.DEFAULT_ANALYZE_FOCUS
            cli_main._render_draft(await engine.analyze(focus))
        elif name == "review":
            if not args:
                _render_pending(engine)
                return
            if len(args) > 1:
                console.print("usage: /review [id]")
                return
            draft_id = _parse_id(args[0])
            if draft_id is None:
                console.print("[red]error:[/red] draft id must be a number")
                return
            _render_review(engine, draft_id)
        elif name == "feedback":
            if len(args) < 2 or _parse_id(args[0]) is None:
                console.print("usage: /feedback <draft_id> <text>")
                return
            draft_id = _parse_id(args[0])
            assert draft_id is not None
            from open_endurance_coach.cli import main as cli_main

            cli_main._render_draft(
                await engine.submit_feedback(draft_id, " ".join(args[1:])), updated=True
            )
        elif name in {"approve", "reject", "apply"}:
            console.print(
                f"[yellow]{name} is confirmation-gated and wired in 2b;"
                f" use `coach {name}` meanwhile.[/yellow]"
            )
    except (LlmError, ValueError, RuntimeError) as exc:
        console.print(f"[red]error:[/red] {exc}")


async def run_chat(engine: CoachEngine) -> None:
    state = ChatState()
    console.print("Chat with the coach. /help lists commands.")
    while True:
        try:
            line = Prompt.ask("[bold cyan]you[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("bye")
            return
        match dispatch(line, state):
            case Ignore():
                continue
            case Exit():
                console.print("bye")
                return
            case Converse():
                console.print(
                    "[dim]Conversation turns are wired in 4c;"
                    " /analyze runs a full review now.[/dim]"
                )
            case UnknownCommand():
                console.print("[red]Unknown command.[/red]")
                console.print(HELP_TEXT)
            case Command(name=name, args=args):
                await _run_command(engine, name, args)


@chat_app.command()
def chat() -> None:
    from open_endurance_coach.cli import main as cli_main

    async def run(engine: CoachEngine) -> None:
        await run_chat(engine)

    cli_main._run(run)

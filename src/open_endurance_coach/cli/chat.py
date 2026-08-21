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
    apply_plan_text,
    console,
    mutations_plan_text,
    reject_plan_text,
    render_apply,
    render_draft,
    render_review,
)
from open_endurance_coach.clients.llm import LlmError
from open_endurance_coach.config import Settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.extractors.deep import detect_deep_query
from open_endurance_coach.store.records import Draft, DraftStatus

chat_app = typer.Typer()

HELP_TEXT = (
    "/analyze [focus]       run a full analysis and save a draft\n"
    "/review [id]           list pending drafts, or inspect one\n"
    "/feedback <id> <text>  answer the coach's questions on a draft\n"
    "/approve <id>          approve a draft (yes/no confirmation)\n"
    "/reject <id>           reject a draft (yes/no confirmation)\n"
    "/apply [id] [--write]  apply decisions (dry-run; --write needs yes)\n"
    "/help                  show this help\n"
    "/exit, /quit           leave the chat\n"
    "Any other line is a conversation turn."
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
    render_review(engine.review(draft_id), chat=True)


def _build_approve_snapshot(engine: CoachEngine, args: list[str]) -> PlanSnapshot | None:
    if len(args) != 1 or _parse_id(args[0]) is None:
        console.print("usage: /approve <draft_id>")
        return None
    draft_id = _parse_id(args[0])
    assert draft_id is not None
    view = engine.review(draft_id)
    if view.draft.status is not DraftStatus.PENDING:
        console.print(
            f"[red]error:[/red] draft #{draft_id} is {view.draft.status.value};"
            " only pending drafts can be approved"
        )
        return None
    return PlanSnapshot(
        action="approve",
        plan_text=mutations_plan_text(draft_id, view.draft.report.mutations),
        draft_id=draft_id,
    )


def _build_reject_snapshot(engine: CoachEngine, args: list[str]) -> PlanSnapshot | None:
    if len(args) != 1 or _parse_id(args[0]) is None:
        console.print("usage: /reject <draft_id>")
        return None
    draft_id = _parse_id(args[0])
    assert draft_id is not None
    view = engine.review(draft_id)
    if view.draft.status is not DraftStatus.PENDING:
        console.print(
            f"[red]error:[/red] draft #{draft_id} is {view.draft.status.value};"
            " only pending drafts can be rejected"
        )
        return None
    return PlanSnapshot(
        action="reject",
        plan_text=reject_plan_text(draft_id),
        draft_id=draft_id,
    )


def _enter_confirmation(snapshot: PlanSnapshot) -> ChatState:
    prompt_plan(snapshot)
    return ChatState(plan=snapshot)


async def _execute_gated(engine: CoachEngine, snapshot: PlanSnapshot) -> None:
    if snapshot.action == "approve":
        assert snapshot.draft_id is not None
        decision = engine.approve(snapshot.draft_id)
        console.print(
            f"Decision #{decision.id} recorded from draft #{snapshot.draft_id}"
            f" ({len(decision.report.mutations)} mutations)."
        )
    elif snapshot.action == "reject":
        assert snapshot.draft_id is not None
        engine.reject(snapshot.draft_id)
        console.print(f"Draft #{snapshot.draft_id} rejected.")
    else:
        report = await engine.apply(snapshot.decision_id, dry_run=not snapshot.write)
        render_apply(report, write=snapshot.write)


async def _handle_converse(engine: CoachEngine, session: ChatSession, text: str) -> None:
    try:
        if session.context is None or detect_deep_query(text) is not None:
            session.context = await engine.build_context(text)
        reply = await engine.converse(text, history=session.history, context=session.context)
        console.print("[bold green]Coach:[/bold green]", end=" ")
        console.print(reply, markup=False)
        session.append(text, reply)
    except (LlmError, ValueError, RuntimeError) as exc:
        console.print(f"[red]error:[/red] {exc}")


async def _handle_confirmation(
    engine: CoachEngine, state: ChatState, line: str, session: ChatSession
) -> ChatState:
    assert state.plan is not None
    snapshot = state.plan

    async def execute(current: CoachEngine) -> None:
        await _execute_gated(current, snapshot)

    async def discuss(line: str) -> None:
        await _handle_converse(engine, session, line)

    async def feedback(line: str, updated: Draft) -> None:
        session.append(line, assistant_turn(updated.report).content)

    try:
        step = await respond(
            engine,
            snapshot,
            line,
            executor=execute,
            chat=True,
            on_discuss=discuss,
            on_feedback=feedback,
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
    try:
        if name == "analyze":
            from open_endurance_coach.cli import main as cli_main

            focus = " ".join(args) if args else cli_main.DEFAULT_ANALYZE_FOCUS
            render_draft(await engine.analyze(focus), chat=True)
            session.context = None
        elif name == "review":
            if not args:
                _render_pending(engine)
                return None
            if len(args) > 1:
                console.print("usage: /review [id]", markup=False)
                return None
            draft_id = _parse_id(args[0])
            if draft_id is None:
                console.print("[red]error:[/red] draft id must be a number")
                return None
            _render_review(engine, draft_id)
        elif name == "feedback":
            if len(args) < 2 or _parse_id(args[0]) is None:
                console.print("usage: /feedback <draft_id> <text>")
                return None
            draft_id = _parse_id(args[0])
            assert draft_id is not None
            text = " ".join(args[1:])
            updated = await engine.submit_feedback(draft_id, text)
            render_draft(updated, updated=True, chat=True)
            session.append(text, assistant_turn(updated.report).content)
        elif name == "approve":
            snapshot = _build_approve_snapshot(engine, args)
            if snapshot is not None:
                return _enter_confirmation(snapshot)
        elif name == "reject":
            snapshot = _build_reject_snapshot(engine, args)
            if snapshot is not None:
                return _enter_confirmation(snapshot)
        elif name == "apply":
            write = "--write" in args
            rest = [arg for arg in args if arg != "--write"]
            if len(rest) > 1 or (rest and _parse_id(rest[0]) is None):
                console.print("usage: /apply [decision_id] [--write]", markup=False)
                return None
            decision_id = _parse_id(rest[0]) if rest else None
            report = await engine.apply(decision_id, dry_run=True)
            if not report.decisions:
                console.print("No unapplied decisions.")
                return None
            if not write:
                render_apply(report, write=False)
                return None
            return _enter_confirmation(
                PlanSnapshot(
                    action="apply",
                    plan_text=f"Decision(s) - write to calendar:\n{apply_plan_text(report)}",
                    draft_id=None,
                    decision_id=decision_id,
                    write=True,
                )
            )
    except (LlmError, ValueError, RuntimeError) as exc:
        console.print(f"[red]error:[/red] {exc}")
    return None


async def run_chat(engine: CoachEngine, settings: Settings) -> None:
    session = ChatSession(cap=settings.chat_history_max_tokens)
    session.seed(
        engine.recent_history(settings.chat_history_turns),
        max_tokens=settings.chat_history_max_tokens,
    )
    state = ChatState()
    console.print("Chat with the coach. /help lists commands.")
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
                await _handle_converse(engine, session, text)
            case UnknownCommand():
                console.print("[red]Unknown command.[/red]")
                console.print(HELP_TEXT, markup=False)
            case Confirmation(line=line):
                state = await _handle_confirmation(engine, state, line, session)
            case Command(name=name, args=args):
                state = await _run_command(engine, name, args, session) or state


@chat_app.command()
def chat() -> None:
    from open_endurance_coach.cli import main as cli_main

    async def run(engine: CoachEngine) -> None:
        await run_chat(engine, cli_main.get_settings())

    cli_main._run(run)

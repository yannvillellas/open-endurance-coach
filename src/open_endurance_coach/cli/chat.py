import re
from dataclasses import dataclass

from pydantic import ValidationError
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
from open_endurance_coach.chat.gate import (
    RECOVERABLE_EXCEPTIONS,
    PlanSnapshot,
    is_exit_command,
)
from open_endurance_coach.chat.history import ChatSession, assistant_turn
from open_endurance_coach.chat.state import ChatState
from open_endurance_coach.cli.confirmation import Done, prompt_plan, respond
from open_endurance_coach.cli.rendering import (
    console,
    escape,
    mutations_plan_text,
    print_error,
    render_apply,
    render_report,
    thinking,
)
from open_endurance_coach.config import Settings
from open_endurance_coach.engine.coach import CoachEngine
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import Mutation
from open_endurance_coach.store.records import Draft

HELP_TEXT = (
    "Just talk to the coach: ask about your training, discuss it, or ask for a plan.\n"
    "When he proposes calendar changes, answer with exactly yes or no (cancel\n"
    "abandons it; anything else is a change request and nothing is written).\n"
    "/provider [name]       show or switch the LLM provider\n"
    "/model [name]          show or switch the LLM model\n"
    "/forget [days]         forget stored history (all, or older than N days)\n"
    "/help                  show this help\n"
    "/exit, /quit           leave the chat\n"
)

_QUESTION_RE = re.compile(r"\b(what|why|how|explain|detail\w*|which|when|who)\b", re.IGNORECASE)
_QUESTION_START_RE = re.compile(
    r"^\s*(?:what|why|how|which|when|who|explain|detail\w*)\b", re.IGNORECASE
)
_CHANGE_RE = re.compile(
    r"\b(make|change|prefer|instead|rather|shorter|longer|less|more|add|remove|modify|adjust|update)\b",
    re.IGNORECASE,
)
_RETRY_RE = re.compile(r"^\s*retry\s*$", re.IGNORECASE)
_BARE_COMMAND_RE = re.compile(
    r"^\s*(help|exit|quit|forget(?:\s+\d+)?|provider(?:\s+\w+)?|model(?:\s+\w+)?)\s*$",
    re.IGNORECASE,
)
_ASSUME_RE = re.compile(
    r"\b(proceed with assumptions|use assumptions|assume|i don'?t know|no idea|not sure yet)\b",
    re.IGNORECASE,
)


def _print_needs_input(questions: list[str]) -> None:
    console.print("[yellow]The coach needs answers before proposing calendar changes:[/yellow]")
    for question in questions:
        console.print(f"  ? {escape(question)}")
    console.print('[dim]Answer here, or say "proceed with assumptions" to plan anyway.[/dim]')


def _handle_llm_command(engine: CoachEngine, name: str, args: list[str]) -> None:
    try:
        if args:
            if name == "provider":
                provider, model = engine.select_llm(provider=args[0])
            else:
                provider, model = engine.select_llm(model=args[0])
        else:
            provider, model = engine.llm_selection()
        console.print(f"Using {escape(provider)} ({escape(model)}).")
    except RECOVERABLE_EXCEPTIONS as exc:
        print_error(exc)


def _open_proposal(draft_id: int, mutations: list[Mutation]) -> ChatState:
    snapshot = PlanSnapshot(
        plan_text="Apply this to Intervals.icu:\n" + mutations_plan_text(mutations),
        draft_id=draft_id,
    )
    return _enter_confirmation(snapshot)


def _enter_confirmation(snapshot: PlanSnapshot) -> ChatState:
    prompt_plan(snapshot)
    return ChatState(plan=snapshot)


async def _analyze_line(engine: CoachEngine, session: ChatSession, focus: str) -> ChatState | None:
    cached = (
        session.context.model_copy(update={"focus": focus}) if session.context is not None else None
    )
    async with thinking():
        draft = await engine.analyze(focus, context=cached, history=session.history)
    report = draft.report
    if report.needs_input:
        blocking = {question.strip().casefold() for question in report.needs_input}
        remaining = [
            question for question in report.questions if question.strip().casefold() not in blocking
        ]
        if len(remaining) != len(report.questions):
            report = report.model_copy(update={"questions": remaining})
    render_report(report)
    session.context = draft.context
    session.append(focus, assistant_turn(draft.report).content)
    needs_input = draft.report.needs_input
    assumed = _ASSUME_RE.search(focus) is not None
    if draft.report.mutations:
        if draft.report.intent != "plan":
            console.print(
                "[dim]The coach drafted calendar changes but did not read this as a"
                " planning request; ask him to plan if you want a proposal.[/dim]"
            )
            return None
        if needs_input and not assumed:
            _print_needs_input(needs_input)
            return None
        return _open_proposal(draft.id, draft.report.mutations)
    if needs_input:
        _print_needs_input(needs_input)
    else:
        console.print("[dim]Answer my questions here if you like.[/dim]")
    return None


async def _retry_apply(engine: CoachEngine, session: ChatSession, text: str) -> None:
    if session.pending_decision_id is None:
        console.print("Nothing to apply.")
        return
    try:
        report = await engine.apply(session.pending_decision_id)
    except RECOVERABLE_EXCEPTIONS as exc:
        print_error(exc)
        return
    session.pending_decision_id = None
    render_apply(report)
    session.append(text, "Applied the recorded calendar changes.")


async def _handle_text(engine: CoachEngine, session: ChatSession, text: str) -> ChatState | None:
    try:
        if _RETRY_RE.match(text):
            await _retry_apply(engine, session, text)
            return None
        return await _analyze_line(engine, session, text)
    except RECOVERABLE_EXCEPTIONS as exc:
        print_error(exc)
        return None


async def _apply_proposal(engine: CoachEngine, session: ChatSession, draft_id: int) -> None:
    decision = engine.approve(draft_id)
    try:
        render_apply(await engine.apply(decision.id))
        session.pending_decision_id = None
    except RECOVERABLE_EXCEPTIONS as exc:
        print_error(exc)
        session.pending_decision_id = decision.id
        console.print(
            f"[yellow]Decision #{decision.id} was recorded but not applied;"
            ' say "retry" to apply it again.[/yellow]'
        )


@dataclass(frozen=True)
class ExitChat:
    pass


async def _handle_proposal(
    engine: CoachEngine, state: ChatState, line: str, session: ChatSession
) -> ChatState | ExitChat:
    assert state.plan is not None
    snapshot = state.plan
    draft_id = snapshot.draft_id
    if is_exit_command(line):
        console.print("[yellow]Cancelled. Nothing changed.[/yellow]")
        return ExitChat()

    if line.startswith("/"):
        parts = line[1:].split()
        name = parts[0].casefold() if parts else ""
        if name == "help":
            console.print(HELP_TEXT, markup=False)
        elif name == "forget":
            console.print("[yellow]/forget is unavailable while a proposal is open.[/yellow]")
        elif name in {"provider", "model"}:
            _handle_llm_command(engine, name, parts[1:])
        else:
            console.print("[red]Unknown command.[/red]")
            console.print(HELP_TEXT, markup=False)
        prompt_plan(snapshot)
        return state

    if _QUESTION_START_RE.search(line) or (
        _QUESTION_RE.search(line) and not _CHANGE_RE.search(line)
    ):
        try:
            draft = engine.review(draft_id)
            context_base = session.context if session.context is not None else draft.context
            try:
                context = CoachContext.model_validate(
                    {
                        **context_base.model_dump(),
                        "current_proposal": draft.report,
                    }
                )
            except ValidationError:
                context = context_base
            async with thinking():
                answer = await engine.analyze(line, context=context, history=session.history)
            render_report(answer.report)
            session.append(line, assistant_turn(answer.report).content)
        except RECOVERABLE_EXCEPTIONS as exc:
            print_error(exc)
        console.print(
            '[dim]Note: to revise the plan, describe the change (e.g. "make it 45 minutes").[/dim]'
        )
        prompt_plan(snapshot)
        return state

    async def execute(current: CoachEngine) -> None:
        await _apply_proposal(current, session, draft_id)

    async def feedback(line: str, updated: Draft) -> bool | None:
        session.append(line, assistant_turn(updated.report).content)
        if not updated.report.mutations:
            console.print("[yellow]No changes proposed anymore.[/yellow]")
            return True
        return None

    def restate(draft: Draft) -> str:
        return "Apply this to Intervals.icu:\n" + mutations_plan_text(draft.report.mutations)

    try:
        step = await respond(
            engine,
            snapshot,
            line,
            executor=execute,
            on_feedback=feedback,
            restate=restate,
        )
    except RECOVERABLE_EXCEPTIONS as exc:
        print_error(exc)
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
    if name == "forget":
        days: int | None = None
        if args:
            if not args[0].isdigit():
                console.print("[red]Usage: /forget [days][/red]")
                return None
            days = int(args[0])
        removed = engine.prune_history(days)
        if days is None:
            session.history = []
            session.context = None
            session.pending_decision_id = None
        scope = "all history" if days is None else f"history older than {days} days"
        console.print(f"Forgot {sum(removed.values())} records ({scope}).")
        return None
    if name in {"provider", "model"}:
        _handle_llm_command(engine, name, args)
        return None
    return None


async def run_chat(engine: CoachEngine, settings: Settings) -> None:
    session = ChatSession(cap=settings.chat_history_max_tokens)
    if settings.history_days > 0:
        removed = engine.prune_history(settings.history_days)
        total = sum(removed.values())
        if total:
            console.print(
                f"[dim]Pruned {total} old records (keeping {settings.history_days} days).[/dim]"
            )
    session.seed(
        engine.recent_history(
            settings.chat_history_turns,
            max_age_days=settings.chat_history_max_age_days,
        ),
        max_tokens=settings.chat_history_max_tokens,
    )
    state = ChatState()
    remembered = sum(1 for turn in session.history if turn.role == "user")
    provider, model = engine.llm_selection()
    console.print("Chat with the coach. /help lists commands.")
    console.print(f"[dim]Using {escape(provider)} ({escape(model)}).[/dim]")
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
        try:
            bare = _BARE_COMMAND_RE.match(line)
            if bare is not None:
                console.print(
                    f"[dim]That looks like a command; type it with a slash:"
                    f" /{bare.group(1).lower()}[/dim]"
                )
                continue
            match dispatch(line, state):
                case Ignore():
                    continue
                case Exit():
                    console.print("bye")
                    return
                case Converse(text=text):
                    state = await _handle_text(engine, session, text) or state
                case UnknownCommand():
                    console.print("[red]Unknown command.[/red]")
                    console.print(HELP_TEXT, markup=False)
                case Confirmation(line=line):
                    step = await _handle_proposal(engine, state, line, session)
                    if isinstance(step, ExitChat):
                        console.print("bye")
                        return
                    state = step
                case Command(name=name, args=args):
                    state = await _run_command(engine, name, args, session) or state
        except Exception as exc:
            print_error(exc)


def start_chat(*, provider: str | None = None, model: str | None = None) -> None:
    from open_endurance_coach.cli import main as cli_main

    async def run(engine: CoachEngine) -> None:
        await run_chat(engine, cli_main.get_settings())

    cli_main._run(run, provider=provider, model=model)

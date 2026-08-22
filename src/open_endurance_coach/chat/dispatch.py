from dataclasses import dataclass

from open_endurance_coach.chat.gate import EXIT_NAMES
from open_endurance_coach.chat.state import ChatMode, ChatState

_COMMANDS = frozenset({"help", "analyze", "clear"})


@dataclass(frozen=True)
class Converse:
    text: str


@dataclass(frozen=True)
class Command:
    name: str
    args: list[str]


@dataclass(frozen=True)
class Confirmation:
    line: str


@dataclass(frozen=True)
class UnknownCommand:
    line: str


@dataclass(frozen=True)
class Exit:
    pass


@dataclass(frozen=True)
class Ignore:
    pass


Action = Converse | Command | Confirmation | UnknownCommand | Exit | Ignore


def dispatch(line: str, state: ChatState) -> Action:
    stripped = line.strip()
    if not stripped:
        return Ignore()
    if state.mode == ChatMode.CONFIRMING:
        return Confirmation(stripped)
    if not stripped.startswith("/"):
        return Converse(stripped)
    tokens = stripped[1:].split()
    name = tokens[0].casefold() if tokens else ""
    if name in EXIT_NAMES:
        return Exit()
    if name in _COMMANDS:
        return Command(name, tokens[1:])
    return UnknownCommand(stripped)

import sqlite3
from dataclasses import dataclass

from open_endurance_coach.clients.llm import LlmError

RECOVERABLE_EXCEPTIONS = (LlmError, ValueError, RuntimeError, sqlite3.Error)

EXIT_NAMES = frozenset({"exit", "quit"})


@dataclass(frozen=True)
class PlanSnapshot:
    plan_text: str
    draft_id: int


@dataclass(frozen=True)
class Proceed:
    pass


@dataclass(frozen=True)
class Declined:
    pass


@dataclass(frozen=True)
class Cancelled:
    pass


@dataclass(frozen=True)
class Ignored:
    pass


@dataclass(frozen=True)
class Feedback:
    line: str


ConfirmationResult = Proceed | Declined | Cancelled | Ignored | Feedback


def is_exit_command(line: str) -> bool:
    words = line.strip().casefold().split()
    return bool(words) and words[0] in {f"/{name}" for name in EXIT_NAMES}


def handle(line: str, snapshot: PlanSnapshot) -> ConfirmationResult:
    stripped = line.strip()
    if not stripped:
        return Ignored()
    key = stripped.casefold()
    if key == "yes":
        return Proceed()
    if key == "no":
        return Declined()
    if key == "cancel":
        return Cancelled()
    return Feedback(stripped)

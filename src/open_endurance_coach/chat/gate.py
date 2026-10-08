import re
import sqlite3
from dataclasses import dataclass
from datetime import date

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.clients.llm import LlmError
from open_endurance_coach.writer.calendar import WriterError

RECOVERABLE_EXCEPTIONS = (LlmError, IntervalsApiError, WriterError, ValueError, sqlite3.Error)

EXIT_NAMES = frozenset({"exit", "quit"})

_SELECTION_RE = re.compile(r"^yes\s+(except\s+)?(\S.*)$")
_SELECTOR_SPLIT_RE = re.compile(r"\s*,\s*|\s+and\s+|\s+")
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass(frozen=True)
class PlanItem:
    """One numbered proposal item: its mutation index and its calendar day, if known."""

    index: int
    day: date | None


@dataclass(frozen=True)
class PlanSnapshot:
    plan_text: str
    proposal_id: int
    items: tuple[PlanItem, ...] = ()


@dataclass(frozen=True)
class Proceed:
    pass


@dataclass(frozen=True)
class ProceedSubset:
    indices: tuple[int, ...]


@dataclass(frozen=True)
class InvalidSelection:
    reason: str


@dataclass(frozen=True)
class Declined:
    pass


@dataclass(frozen=True)
class Ignored:
    pass


@dataclass(frozen=True)
class Feedback:
    line: str


ConfirmationResult = Proceed | ProceedSubset | InvalidSelection | Declined | Ignored | Feedback


def is_exit_command(line: str) -> bool:
    words = line.strip().casefold().split()
    return bool(words) and words[0] in {f"/{name}" for name in EXIT_NAMES}


def _selector_matches(token: str, items: tuple[PlanItem, ...]) -> set[int] | None:
    """Item positions (0-based) a selector picks, or None when it is not a selector."""
    if token.isdigit():
        if not token.isascii():
            return set()
        try:
            number = int(token)
        except ValueError:
            return set()
        return {number - 1} if 1 <= number <= len(items) else set()
    weekday = next(
        (day for day, name in enumerate(_WEEKDAYS) if len(token) >= 3 and name.startswith(token)),
        None,
    )
    if weekday is not None:
        return {
            position
            for position, item in enumerate(items)
            if item.day is not None and item.day.weekday() == weekday
        }
    try:
        day = date.fromisoformat(token)
    except ValueError:
        return None
    return {position for position, item in enumerate(items) if item.day == day}


def _select(key: str, snapshot: PlanSnapshot) -> ConfirmationResult | None:
    """Resolve "yes <items>" / "yes except <items>"; None when the line is not a selection."""
    match = _SELECTION_RE.match(key)
    if match is None:
        return None
    items = snapshot.items
    tokens = [token for token in _SELECTOR_SPLIT_RE.split(match.group(2)) if token]
    if not tokens:
        return InvalidSelection("No items were given.")
    picked: set[int] = set()
    for token in tokens:
        matches = _selector_matches(token, items)
        if matches is None:
            return None
        if not matches:
            return InvalidSelection(f"No proposed item matches {token}.")
        picked |= matches
    if match.group(1):
        picked = set(range(len(items))) - picked
    if not picked:
        return InvalidSelection("That leaves nothing to approve.")
    if len(picked) == len(items):
        return Proceed()
    return ProceedSubset(tuple(sorted(items[position].index for position in picked)))


def handle(line: str, snapshot: PlanSnapshot) -> ConfirmationResult:
    stripped = line.strip()
    if not stripped:
        return Ignored()
    key = stripped.casefold()
    if key == "yes":
        return Proceed()
    if key == "no":
        return Declined()
    selection = _select(key, snapshot)
    if selection is not None:
        return selection
    return Feedback(stripped)

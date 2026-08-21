from dataclasses import dataclass
from typing import Literal

ConfirmationAction = Literal["approve", "apply", "reject"]


@dataclass(frozen=True)
class PlanSnapshot:
    action: ConfirmationAction
    plan_text: str
    draft_id: int | None
    decision_id: int | None = None
    write: bool = False


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


@dataclass(frozen=True)
class Discuss:
    line: str


ConfirmationResult = Proceed | Declined | Cancelled | Ignored | Feedback | Discuss


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
    if snapshot.action == "apply" or snapshot.draft_id is None:
        return Discuss(stripped)
    return Feedback(stripped)

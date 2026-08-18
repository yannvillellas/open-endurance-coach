from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport


class DraftStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Draft:
    id: int
    created_at: datetime
    status: DraftStatus
    focus: str
    user_feedback: str | None
    context: CoachContext
    report: DecisionReport


@dataclass(frozen=True)
class Feedback:
    id: int
    draft_id: int
    created_at: datetime
    content: str


@dataclass(frozen=True)
class Decision:
    id: int
    draft_id: int
    decided_at: datetime
    applied_at: datetime | None
    report: DecisionReport

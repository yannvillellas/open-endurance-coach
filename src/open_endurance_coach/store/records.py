from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport


class ProposalStatus(StrEnum):
    PENDING = "pending"
    UNAPPLIED = "unapplied"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Proposal:
    id: int
    created_at: datetime
    status: ProposalStatus
    focus: str
    user_feedback: str | None
    context: CoachContext
    report: DecisionReport
    approved_report: DecisionReport | None = None
    decided_at: datetime | None = None


@dataclass(frozen=True)
class Message:
    id: int
    created_at: datetime
    role: MessageRole
    content: str
    report: DecisionReport | None = None

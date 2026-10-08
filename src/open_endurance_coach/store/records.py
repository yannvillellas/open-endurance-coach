from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport


class ProposalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


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
    applied_at: datetime | None = None


@dataclass(frozen=True)
class Feedback:
    id: int
    proposal_id: int
    created_at: datetime
    content: str


@dataclass(frozen=True)
class FeedbackWithReport:
    feedback: Feedback
    report: DecisionReport

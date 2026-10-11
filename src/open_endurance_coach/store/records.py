from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport


class ProposalStatus(StrEnum):
    PENDING = "pending"
    UNAPPLIED = "unapplied"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class MemoryCategory(StrEnum):
    ATHLETE_PROFILE = "athlete_profile"
    GOALS_MOTIVATION = "goals_motivation"
    HEALTH_MEDICAL = "health_medical"
    INJURY_NIGGLE = "injury_niggle"
    TRAINING_CONSTRAINTS = "training_constraints"
    PREFERENCES_TOLERANCE = "preferences_tolerance"
    OBSERVED_PATTERNS = "observed_patterns"


HISTORY_CATEGORIES = frozenset({MemoryCategory.HEALTH_MEDICAL, MemoryCategory.INJURY_NIGGLE})


class MemoryOrigin(StrEnum):
    DECLARED = "declared"
    OBSERVED = "observed"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"


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


@dataclass(frozen=True)
class MemoryFact:
    id: int
    created_at: datetime
    updated_at: datetime
    category: MemoryCategory
    origin: MemoryOrigin
    text: str
    status: MemoryStatus
    valid_from: date
    valid_to: date | None
    expires_on: date | None
    trigger_keywords: tuple[str, ...]
    trigger_on: date | None
    last_surfaced_at: datetime | None
    source_message_id: int | None

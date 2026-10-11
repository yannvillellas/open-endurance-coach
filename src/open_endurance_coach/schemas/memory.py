from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


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


class MemoryAction(StrEnum):
    REMEMBER = "remember"
    UPDATE = "update"
    CLOSE = "close"
    FORGET = "forget"


class MemoryItem(BaseModel):
    """One active fact shown in full to the model."""

    model_config = ConfigDict(extra="forbid")

    id: int
    category: MemoryCategory
    origin: MemoryOrigin
    text: str
    valid_from: date
    expires_on: date | None = None


class MemoryIndexEntry(BaseModel):
    """One active fact beyond the injection cap, shown as a compact index row."""

    model_config = ConfigDict(extra="forbid")

    id: int
    category: MemoryCategory
    label: str

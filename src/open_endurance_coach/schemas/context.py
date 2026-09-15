from datetime import date
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_endurance_coach.schemas.decisions import DecisionReport, RaceCategory
from open_endurance_coach.schemas.intervals import Activity, Event, SportSettings, Wellness
from open_endurance_coach.tokens import estimate_payload_tokens

MacroPhase = Literal["Base", "Build", "Peak", "Taper", "Race week"]


def _tokens_of(payload: Any) -> int:
    return estimate_payload_tokens(payload)


_SECTION_KEYS = (
    "focus",
    "today",
    "current_proposal",
    "recent_activities",
    "activity_detail",
    "wellness",
    "upcoming_events",
    "goal_races",
    "training_rollup",
    "sport_settings",
    "user_feedback",
)


class SportWeek(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1)
    sessions: int = Field(ge=0)
    time_s: int = Field(ge=0)
    load: float | None = None


class TrainingWeek(BaseModel):
    model_config = ConfigDict(extra="forbid")

    week_start: date
    partial: bool = False
    sessions: int = Field(ge=0)
    time_s: int = Field(ge=0)
    load: float | None = None
    fitness: float | None = None
    fatigue: float | None = None
    form: float | None = None
    ramp_rate: float | None = None
    sports: list[SportWeek] = Field(default_factory=list)


class GoalRace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: int | str | None = None
    name: str = Field(min_length=1)
    date: date
    category: RaceCategory
    type: str | None = None
    days_to_race: int = Field(ge=0)
    weeks_to_race: int = Field(ge=0)
    phase: MacroPhase


class CoachContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus: str = Field(min_length=1)
    today: date | None = None
    current_proposal: DecisionReport | None = None
    recent_activities: list[Activity] = Field(default_factory=list)
    activity_detail: Activity | None = None
    wellness: list[Wellness] = Field(default_factory=list)
    upcoming_events: list[Event] = Field(default_factory=list)
    goal_races: list[GoalRace] = Field(default_factory=list)
    training_rollup: list[TrainingWeek] = Field(default_factory=list)
    sport_settings: list[SportSettings] = Field(default_factory=list)
    user_feedback: str | None = None
    max_tokens: int = Field(default=8192, gt=0)

    def sections(self) -> dict[str, Any]:
        sections: dict[str, Any] = {
            "focus": self.focus,
            "recent_activities": [
                item.model_dump(mode="json", exclude_none=True) for item in self.recent_activities
            ],
            "activity_detail": (
                self.activity_detail.model_dump(mode="json", exclude_none=True)
                if self.activity_detail
                else None
            ),
            "wellness": [item.model_dump(mode="json", exclude_none=True) for item in self.wellness],
            "upcoming_events": [
                item.model_dump(mode="json", exclude_none=True) for item in self.upcoming_events
            ],
            "goal_races": [
                item.model_dump(mode="json", exclude_none=True) for item in self.goal_races
            ],
            "training_rollup": [
                item.model_dump(mode="json", exclude_none=True) for item in self.training_rollup
            ],
            "sport_settings": [
                item.model_dump(mode="json", exclude_none=True) for item in self.sport_settings
            ],
        }
        if self.today:
            sections["today"] = f"Today's date (athlete local): {self.today.isoformat()}"
        if self.current_proposal:
            sections["current_proposal"] = self.current_proposal.model_dump(
                mode="json", exclude_none=True
            )
        if self.user_feedback:
            sections["user_feedback"] = self.user_feedback
        return sections

    def section_tokens(self) -> dict[str, int]:
        sections = self.sections()
        return {key: _tokens_of(sections[key]) if key in sections else 0 for key in _SECTION_KEYS}

    def estimated_tokens(self) -> int:
        return max(1, sum(self.section_tokens().values()))

    @model_validator(mode="after")
    def _within_budget(self) -> Self:
        estimated = self.estimated_tokens()
        if estimated > self.max_tokens:
            raise ValueError(f"context exceeds token budget: {estimated} > {self.max_tokens}")
        return self

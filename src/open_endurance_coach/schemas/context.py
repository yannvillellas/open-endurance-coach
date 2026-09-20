from datetime import date
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_endurance_coach.schemas.decisions import DecisionReport, RaceCategory
from open_endurance_coach.schemas.intervals import (
    Activity,
    ActivitySplit,
    Event,
    SportSettings,
    Wellness,
)
from open_endurance_coach.tokens import estimate_payload_tokens, estimate_text_tokens

MacroPhase = Literal["Base", "Build", "Peak", "Taper", "Race week"]


def _tokens_of(payload: Any) -> int:
    if payload is None or payload == "" or payload == [] or payload == {}:
        return 0
    return estimate_payload_tokens(payload)


_SECTION_KEYS = (
    "focus",
    "recent_activities",
    "activity_detail",
    "wellness",
    "recent_events",
    "upcoming_events",
    "goal_races",
    "training_rollup",
    "sport_settings",
    "activity_splits",
    "today",
    "current_proposal",
    "user_feedback",
)

# Events are projected to the fields the model needs: ``workout_doc`` and the plan
# ids are large and unused, while ``id`` must stay so mutations can reference it.
_EVENT_FIELDS = (
    "id",
    "name",
    "start_date_local",
    "category",
    "type",
    "description",
    "end_date_local",
    "moving_time",
    "distance",
    "icu_training_load",
)


def _event_payload(event: Event) -> dict[str, Any]:
    return event.model_dump(mode="json", include=set(_EVENT_FIELDS), exclude_none=True)


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
    moving_time: int | None = None
    distance: float | None = None
    icu_training_load: float | None = None


class CoachContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus: str = Field(min_length=1)
    today: date | None = None
    current_proposal: DecisionReport | None = None
    recent_activities: list[Activity] = Field(default_factory=list)
    activity_detail: Activity | None = None
    activity_splits: list[ActivitySplit] = Field(default_factory=list)
    wellness: list[Wellness] = Field(default_factory=list)
    recent_events: list[Event] = Field(default_factory=list)
    upcoming_events: list[Event] = Field(default_factory=list)
    goal_races: list[GoalRace] = Field(default_factory=list)
    training_rollup: list[TrainingWeek] = Field(default_factory=list)
    sport_settings: list[SportSettings] = Field(default_factory=list)
    user_feedback: str | None = None
    max_tokens: int = Field(default=16384, gt=0)

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
            "recent_events": [_event_payload(event) for event in self.recent_events],
            "upcoming_events": [_event_payload(event) for event in self.upcoming_events],
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
        if self.activity_splits:
            sections["activity_splits"] = [
                item.model_dump(mode="json", exclude_none=True) for item in self.activity_splits
            ]
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
        """Per-section token estimates, for diagnostics only.

        The budget is measured on the serialized payload the prompt actually sends
        (see ``data_tokens``); summing these per-section dumps undercounts it because
        the prompt nests every section one level deeper.
        """
        sections = self.sections()
        return {key: _tokens_of(sections[key]) if key in sections else 0 for key in _SECTION_KEYS}

    def data_payload(self) -> dict[str, Any]:
        """The athlete-data dict the prompt serializes, without the message."""
        return {key: value for key, value in self.sections().items() if key != "focus"}

    def data_tokens(self) -> int:
        """Tokens of athlete data as the prompt renders it, excluding the message."""
        return _tokens_of(self.data_payload())

    def estimated_tokens(self) -> int:
        """Tokens of the data payload plus the athlete's message."""
        return max(1, self.data_tokens() + estimate_text_tokens(self.focus))

    @model_validator(mode="after")
    def _within_budget(self) -> Self:
        data = self.data_tokens()
        if data > self.max_tokens:
            raise ValueError(f"context data exceeds token budget: {data} > {self.max_tokens}")
        return self

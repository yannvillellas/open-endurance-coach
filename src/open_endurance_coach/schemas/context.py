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
from open_endurance_coach.schemas.memory import MemoryIndexEntry, MemoryItem
from open_endurance_coach.tokens import estimate_payload_tokens, estimate_text_tokens

BlockPhase = Literal["Prep", "Base", "Build", "Peak", "Race"]
BlockWeekStatus = Literal[
    "done",
    "under",
    "over",
    "current",
    "upcoming",
    "rest",
    "repeat",
    "step_back",
    "omitted",
]


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
    "block_plan",
    "execution_review",
    "training_rollup",
    "sport_settings",
    "activity_splits",
    "today",
    "current_proposal",
    "user_feedback",
    "coach_memory",
    "coach_memory_index",
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


_MODEL_LIST_KEYS = frozenset(
    {
        "recent_activities",
        "wellness",
        "goal_races",
        "training_rollup",
        "sport_settings",
        "activity_splits",
        "coach_memory",
        "coach_memory_index",
    }
)


def _section_payload(key: str, value: Any) -> Any:
    """The rendered value of one section, shared by ``sections`` and budgeting."""
    if key in ("recent_events", "upcoming_events"):
        return [_event_payload(event) for event in value]
    if key == "block_plan":
        return block_plan_table(value)
    if key == "execution_review":
        return execution_review_text(value)
    if key in _MODEL_LIST_KEYS:
        return [item.model_dump(mode="json", exclude_none=True) for item in value]
    if key in ("activity_detail", "current_proposal"):
        return value.model_dump(mode="json", exclude_none=True) if value else None
    return value


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
    moving_time: int | None = None
    distance: float | None = None
    description: str | None = None
    description_truncated: bool | None = None
    icu_training_load: float | None = None


class BlockWeek(BaseModel):
    """One week of the race block: phase, role, target load band and status."""

    model_config = ConfigDict(extra="forbid")

    week_start: date
    phase: BlockPhase
    week_in_phase: int = Field(ge=1)
    role: str = Field(min_length=1)
    target_tss_low: float | None = None
    target_tss_high: float | None = None
    status: BlockWeekStatus


def _table_cell(text: str) -> str:
    """One single-line cell: untrusted role text can neither break nor forge rows."""
    return " ".join(text.split()).replace("|", "\\|")


def block_plan_table(rows: list[BlockWeek]) -> str:
    """The block plan as a compact Markdown table, one row per week."""
    if not rows:
        return ""
    lines = [
        "| Week | Phase | Role | Target | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        phase = row.phase if row.phase == "Race" else f"{row.phase} {row.week_in_phase}"
        if row.target_tss_low is None or row.target_tss_high is None:
            target = "—"
        else:
            target = f"{row.target_tss_low:.0f}-{row.target_tss_high:.0f}"
        lines.append(
            f"| {row.week_start.isoformat()} | {phase} | {_table_cell(row.role)}"
            f" | {target} | {row.status} |"
        )
    return "\n".join(lines)


class ReviewWeek(BaseModel):
    """One judged week of the execution review, joined with its adjusted plan row."""

    model_config = ConfigDict(extra="forbid")

    week_start: date
    phase: BlockPhase
    status: BlockWeekStatus
    band_low: float | None = None
    band_high: float | None = None
    planned_load: float | None = None
    executed_load: float | None = None
    completed: int = Field(default=0, ge=0)
    partial: int = Field(default=0, ge=0)
    missed: int = Field(default=0, ge=0)
    adherence: float | None = None
    unplanned_load: float | None = None
    unplanned_share: float | None = None
    fitness: float | None = None
    ramp: float | None = None
    current: bool = False


class ReviewSession(BaseModel):
    """One judged session (or unplanned activity) with its exact planned/actual numbers."""

    model_config = ConfigDict(extra="forbid")

    date: date
    status: Literal["completed", "partial", "missed", "unplanned"]
    planned: str | None = None
    planned_load: float | None = None
    planned_time_s: int | None = None
    planned_distance_m: float | None = None
    actual: str | None = None
    actual_load: float | None = None
    actual_time_s: int | None = None
    actual_distance_m: float | None = None
    compliance: float | None = None


class ExecutionReview(BaseModel):
    """The compact execution digest injected for a review turn."""

    model_config = ConfigDict(extra="forbid")

    weeks: list[ReviewWeek] = Field(default_factory=list)
    sessions: list[ReviewSession] = Field(default_factory=list)


def _quantity(load: float | None, time_s: int | None, distance_m: float | None) -> str:
    if load is not None:
        return f"{load:.0f} TSS"
    if time_s is not None:
        return f"{time_s / 3600:.1f} h"
    if distance_m is not None:
        return f"{distance_m / 1000:.1f} km"
    return "—"


def execution_review_text(review: ExecutionReview) -> str:
    """The execution review as a compact Markdown table plus one line per session."""
    if not review.weeks and not review.sessions:
        return ""
    lines: list[str] = []
    if review.weeks:
        lines.extend(
            (
                "| Week | Phase | Status | Band | Planned | Executed | C/P/M | Adherence"
                " | Unplanned | CTL | Ramp |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            )
        )
    for week in review.weeks:
        band = (
            "—"
            if week.band_low is None or week.band_high is None
            else f"{week.band_low:.0f}-{week.band_high:.0f}"
        )
        planned = "—" if week.planned_load is None else f"{week.planned_load:.0f}"
        executed = "—" if week.executed_load is None else f"{week.executed_load:.0f}"
        adherence = "—" if week.adherence is None else f"{week.adherence:.0%}"
        fitness = "—" if week.fitness is None else f"{week.fitness:.0f}"
        ramp = "—" if week.ramp is None else f"{week.ramp:+.1f}"
        if week.unplanned_load is None:
            unplanned = "—"
        elif week.unplanned_share is None:
            unplanned = f"{week.unplanned_load:.0f}"
        else:
            unplanned = f"{week.unplanned_load:.0f} ({week.unplanned_share:.0%})"
        lines.append(
            f"| {week.week_start.isoformat()} | {week.phase} | {week.status} | {band}"
            f" | {planned} | {executed} | {week.completed}/{week.partial}/{week.missed}"
            f" | {adherence} | {unplanned} | {fitness} | {ramp} |"
        )
    for session in review.sessions:
        actual_q = _quantity(session.actual_load, session.actual_time_s, session.actual_distance_m)
        name = _table_cell(session.actual or "")
        if session.status == "unplanned":
            lines.append(f"- {session.date.isoformat()} unplanned: {name} {actual_q}")
            continue
        planned_q = _quantity(
            session.planned_load, session.planned_time_s, session.planned_distance_m
        )
        planned = f"{_table_cell(session.planned or '')} {planned_q}"
        if session.status == "missed":
            lines.append(f"- {session.date.isoformat()} planned {planned} — missed")
            continue
        share = f", compliance {session.compliance:.0f}%" if session.compliance is not None else ""
        lines.append(
            f"- {session.date.isoformat()} planned {planned} → {name} {actual_q}"
            f"{share} — {session.status}"
        )
    return "\n".join(lines)


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
    block_plan: list[BlockWeek] = Field(default_factory=list)
    execution_review: ExecutionReview | None = None
    training_rollup: list[TrainingWeek] = Field(default_factory=list)
    sport_settings: list[SportSettings] = Field(default_factory=list)
    user_feedback: str | None = None
    coach_memory: list[MemoryItem] = Field(default_factory=list)
    coach_memory_index: list[MemoryIndexEntry] = Field(default_factory=list)
    max_tokens: int = Field(default=16384, gt=0)

    def _raw_sections(self) -> dict[str, Any]:
        raw: dict[str, Any] = {
            "focus": self.focus,
            "recent_activities": self.recent_activities,
            "activity_detail": self.activity_detail,
            "wellness": self.wellness,
            "recent_events": self.recent_events,
            "upcoming_events": self.upcoming_events,
            "goal_races": self.goal_races,
        }
        if self.block_plan:
            raw["block_plan"] = self.block_plan
        if self.execution_review is not None:
            raw["execution_review"] = self.execution_review
        raw["training_rollup"] = self.training_rollup
        raw["sport_settings"] = self.sport_settings
        if self.activity_splits:
            raw["activity_splits"] = self.activity_splits
        if self.today:
            raw["today"] = f"Today's date (athlete local): {self.today.isoformat()}"
        if self.current_proposal:
            raw["current_proposal"] = self.current_proposal
        if self.user_feedback:
            raw["user_feedback"] = self.user_feedback
        if self.coach_memory:
            raw["coach_memory"] = self.coach_memory
        if self.coach_memory_index:
            raw["coach_memory_index"] = self.coach_memory_index
        return raw

    def sections(self) -> dict[str, Any]:
        return {key: _section_payload(key, value) for key, value in self._raw_sections().items()}

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

import hashlib
import json
import re
from datetime import date
from typing import Annotated, Literal, Self, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from open_endurance_coach.sanitize import sanitize_text
from open_endurance_coach.schemas.memory import MemoryAction, MemoryCategory, MemoryOrigin

_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_event_id(value: int | str) -> int | str:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("event_id must be a positive id from the athlete data")
        return value
    stripped = value.strip()
    if stripped.lstrip("+-").isdigit() and int(stripped) <= 0:
        raise ValueError("event_id must be a positive id from the athlete data")
    if _EVENT_ID_RE.fullmatch(stripped) is None:
        raise ValueError("event_id must be a safe id from the athlete data")
    return stripped


EventId = Annotated[int | str, AfterValidator(_validate_event_id)]


def _validate_name(value: str) -> str:
    """Mutation names are single-line printable text: displayed as they are written."""
    if "\n" in value or "\t" in value or sanitize_text(value) != value:
        raise ValueError("name must be a single line without control characters")
    name = value.strip()
    if not name:
        raise ValueError("name must not be blank")
    return name


MutationName = Annotated[str, AfterValidator(_validate_name)]

# Free text written to the hub: control characters are replaced, not rejected,
# so the displayed plan stays exactly what will be written.
MutationText = Annotated[str, AfterValidator(sanitize_text)]


class CreateWorkout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["create"]
    name: MutationName
    start_date_local: date
    description: MutationText | None = None
    type: MutationText | None = None
    moving_time: int | None = None
    distance: float | None = None
    icu_training_load: float | None = None


class UpdateWorkout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["update"]
    event_id: EventId
    name: MutationName | None = None
    start_date_local: date | None = None
    description: MutationText | None = None
    type: MutationText | None = None
    moving_time: int | None = None
    distance: float | None = None
    icu_training_load: float | None = None

    @model_validator(mode="after")
    def _at_least_one_change(self) -> Self:
        if all(
            value is None
            for value in (
                self.name,
                self.start_date_local,
                self.description,
                self.type,
                self.moving_time,
                self.distance,
                self.icu_training_load,
            )
        ):
            raise ValueError("update mutation must change at least one field")
        return self


class DeleteWorkout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["delete"]
    event_id: EventId


RaceCategory = Literal["RACE_A", "RACE_B", "RACE_C"]
RACE_CATEGORIES: tuple[RaceCategory, ...] = get_args(RaceCategory)


class CreateRace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["create_race"]
    name: MutationName
    start_date_local: date
    category: RaceCategory
    description: MutationText | None = None
    type: MutationText | None = None
    moving_time: int | None = None
    distance: float | None = None
    icu_training_load: float | None = None


class UpdateRace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["update_race"]
    event_id: EventId
    name: MutationName | None = None
    start_date_local: date | None = None
    category: RaceCategory | None = None
    description: MutationText | None = None
    type: MutationText | None = None
    moving_time: int | None = None
    distance: float | None = None
    icu_training_load: float | None = None

    @model_validator(mode="after")
    def _at_least_one_change(self) -> Self:
        if all(
            value is None
            for value in (
                self.name,
                self.start_date_local,
                self.category,
                self.description,
                self.type,
                self.moving_time,
                self.distance,
                self.icu_training_load,
            )
        ):
            raise ValueError("update mutation must change at least one field")
        return self


class DeleteRace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["delete_race"]
    event_id: EventId


Mutation = Annotated[
    CreateWorkout | UpdateWorkout | DeleteWorkout | CreateRace | UpdateRace | DeleteRace,
    Field(discriminator="action"),
]


def _validate_memory_text(value: str) -> str:
    if "\n" in value or "\t" in value or sanitize_text(value) != value:
        raise ValueError("memory text must be a single line without control characters")
    text = value.strip()
    if not text:
        raise ValueError("memory text must not be blank")
    return text


MemoryText = Annotated[str, AfterValidator(_validate_memory_text)]


class MemoryOp(BaseModel):
    """One coach-memory operation, applied outside the calendar approval gate."""

    model_config = ConfigDict(extra="forbid")

    action: MemoryAction
    id: int | None = None
    category: MemoryCategory | None = None
    origin: MemoryOrigin | None = None
    text: MemoryText | None = None
    expires_on: date | None = None
    trigger_keywords: list[str] = Field(default_factory=list)
    trigger_on: date | None = None

    @model_validator(mode="after")
    def _check_action_fields(self) -> Self:
        if self.action is MemoryAction.REMEMBER:
            if self.id is not None:
                raise ValueError("remember must not carry an id")
            if self.category is None or self.origin is None:
                raise ValueError("remember requires category and origin")
            if self.text is None:
                raise ValueError("remember requires text")
            return self
        if self.id is None:
            raise ValueError(f"{self.action.value} requires an id")
        if self.category is not None or self.origin is not None:
            raise ValueError(f"{self.action.value} must not carry category or origin")
        if self.action is MemoryAction.UPDATE:
            if self.text is None:
                raise ValueError("update requires text")
            return self
        if (
            self.text is not None
            or self.expires_on is not None
            or self.trigger_keywords
            or self.trigger_on is not None
        ):
            raise ValueError(f"{self.action.value} must not carry fact content")
        return self


class DecisionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["chat", "analysis", "plan"] = "analysis"
    summary: str = Field(min_length=1)
    findings: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    needs_input: list[str] = Field(default_factory=list)
    mutations: list[Mutation] = Field(default_factory=list)
    memory: list[MemoryOp] = Field(default_factory=list)


def mutations_fingerprint(report: DecisionReport) -> str:
    """Stable content hash binding an approval to the plan that was displayed."""
    mutations = report.model_dump(mode="json")["mutations"]
    payload = json.dumps(mutations, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

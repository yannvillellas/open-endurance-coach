import re
from datetime import date
from typing import Annotated, Literal, Self, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from open_endurance_coach.sanitize import sanitize_text

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
    name = value.strip()
    if not name:
        raise ValueError("name must not be blank")
    if "\n" in name or "\t" in name or sanitize_text(name) != name:
        raise ValueError("name must be a single line without control characters")
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


class DecisionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["chat", "analysis", "plan"] = "analysis"
    summary: str = Field(min_length=1)
    findings: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    needs_input: list[str] = Field(default_factory=list)
    mutations: list[Mutation] = Field(default_factory=list)

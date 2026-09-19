import re
from datetime import date
from typing import Annotated, Literal, Self, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

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


class CreateWorkout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["create"]
    name: str = Field(min_length=1)
    start_date_local: date
    description: str | None = None
    type: str | None = None
    moving_time: int | None = None
    distance: float | None = None
    icu_training_load: float | None = None


class UpdateWorkout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["update"]
    event_id: EventId
    name: str | None = None
    start_date_local: date | None = None
    description: str | None = None
    type: str | None = None
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
    name: str = Field(min_length=1)
    start_date_local: date
    category: RaceCategory
    description: str | None = None
    type: str | None = None
    moving_time: int | None = None
    distance: float | None = None
    icu_training_load: float | None = None


class UpdateRace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["update_race"]
    event_id: EventId
    name: str | None = None
    start_date_local: date | None = None
    category: RaceCategory | None = None
    description: str | None = None
    type: str | None = None
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

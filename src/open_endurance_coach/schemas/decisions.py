from datetime import date
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CreateWorkout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["create"]
    name: str = Field(min_length=1)
    start_date_local: date
    description: str | None = None
    type: str | None = None
    moving_time: int | None = None
    icu_training_load: float | None = None


class UpdateWorkout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["update"]
    event_id: int | str
    name: str | None = None
    start_date_local: date | None = None
    description: str | None = None
    type: str | None = None
    moving_time: int | None = None
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
                self.icu_training_load,
            )
        ):
            raise ValueError("update mutation must change at least one field")
        return self


class DeleteWorkout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["delete"]
    event_id: int | str


WorkoutMutation = Annotated[
    CreateWorkout | UpdateWorkout | DeleteWorkout, Field(discriminator="action")
]


class DecisionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    findings: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    mutations: list[WorkoutMutation] = Field(default_factory=list)

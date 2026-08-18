import json
from datetime import date
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_endurance_coach.schemas.intervals import Activity, Event, SportSettings, Wellness


def _tokens_of(payload: Any) -> int:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    return len(serialized) // 4


class CoachContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus: str = Field(min_length=1)
    today: date | None = None
    recent_activities: list[Activity] = Field(default_factory=list)
    activity_detail: Activity | None = None
    wellness: list[Wellness] = Field(default_factory=list)
    upcoming_events: list[Event] = Field(default_factory=list)
    sport_settings: list[SportSettings] = Field(default_factory=list)
    user_feedback: str | None = None
    max_tokens: int = Field(default=4096, gt=0)

    def _section_payloads(self) -> dict[str, Any]:
        return {
            "focus": self.focus,
            "today": self.today.isoformat() if self.today else 0,
            "recent_activities": [
                item.model_dump(mode="json", exclude_none=True) for item in self.recent_activities
            ],
            "activity_detail": (
                self.activity_detail.model_dump(mode="json", exclude_none=True)
                if self.activity_detail
                else 0
            ),
            "wellness": [item.model_dump(mode="json", exclude_none=True) for item in self.wellness],
            "upcoming_events": [
                item.model_dump(mode="json", exclude_none=True) for item in self.upcoming_events
            ],
            "sport_settings": [
                item.model_dump(mode="json", exclude_none=True) for item in self.sport_settings
            ],
            "user_feedback": self.user_feedback if self.user_feedback else 0,
        }

    def section_tokens(self) -> dict[str, int]:
        return {name: _tokens_of(payload) for name, payload in self._section_payloads().items()}

    def estimated_tokens(self) -> int:
        return max(1, sum(self.section_tokens().values()))

    @model_validator(mode="after")
    def _within_budget(self) -> Self:
        estimated = self.estimated_tokens()
        if estimated > self.max_tokens:
            raise ValueError(f"context exceeds token budget: {estimated} > {self.max_tokens}")
        return self

from datetime import date
from typing import Any

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.intervals import Activity, Event, SportSettings, Wellness


def build_within_budget(
    focus: str,
    recent_activities: list[Activity],
    wellness: list[Wellness],
    upcoming_events: list[Event],
    sport_settings: list[SportSettings],
    *,
    user_feedback: str | None,
    activity_detail: Activity | None,
    max_tokens: int,
    today: date | None = None,
) -> CoachContext:
    activities = list(recent_activities)
    wellness_rows = list(wellness)
    events = list(upcoming_events)
    while True:
        payload: dict[str, Any] = {
            "focus": focus,
            "today": today,
            "recent_activities": activities,
            "activity_detail": activity_detail,
            "wellness": wellness_rows,
            "upcoming_events": events,
            "sport_settings": sport_settings,
            "user_feedback": user_feedback,
            "max_tokens": max_tokens,
        }
        probe = CoachContext.model_construct(**payload)
        if probe.estimated_tokens() <= max_tokens:
            return CoachContext.model_validate(payload)
        if activities:
            activities.pop()
        elif wellness_rows:
            wellness_rows.pop()
        elif events:
            events.pop()
        else:
            raise RuntimeError(f"cannot fit focus in token budget: {max_tokens}")

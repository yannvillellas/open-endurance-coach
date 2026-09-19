from datetime import date, timedelta
from typing import Any

from open_endurance_coach.schemas.context import CoachContext, GoalRace, TrainingWeek
from open_endurance_coach.schemas.decisions import DecisionReport
from open_endurance_coach.schemas.intervals import (
    Activity,
    ActivitySplit,
    Event,
    SportSettings,
    Wellness,
)

RECENT_ACTIVITY_KEEP_DAYS = 7


def _activity_droppable(activities: list[Activity], today: date | None) -> bool:
    if not activities:
        return False
    if today is None:
        return True
    oldest = min(activity.start_date_local.date() for activity in activities)
    return oldest < today - timedelta(days=RECENT_ACTIVITY_KEEP_DAYS)


def build_within_budget(
    focus: str,
    recent_activities: list[Activity],
    wellness: list[Wellness],
    upcoming_events: list[Event],
    sport_settings: list[SportSettings],
    *,
    goal_races: list[GoalRace] | None = None,
    training_rollup: list[TrainingWeek] | None = None,
    recent_events: list[Event] | None = None,
    activity_keep_ids: set[str] | None = None,
    current_proposal: DecisionReport | None = None,
    user_feedback: str | None,
    activity_detail: Activity | None,
    activity_splits: list[ActivitySplit] | None = None,
    max_tokens: int,
    today: date | None = None,
) -> CoachContext:
    activities = list(recent_activities)
    splits = list(activity_splits or [])
    wellness_rows = list(wellness)
    events = list(upcoming_events)
    past_events = list(recent_events or [])
    races = list(goal_races or [])
    rollup = list(training_rollup or [])
    keep_ids = activity_keep_ids or set()
    while True:
        payload: dict[str, Any] = {
            "focus": focus,
            "today": today,
            "recent_activities": activities,
            "activity_detail": activity_detail,
            "activity_splits": splits,
            "wellness": wellness_rows,
            "recent_events": past_events,
            "upcoming_events": events,
            "goal_races": races,
            "training_rollup": rollup,
            "sport_settings": sport_settings,
            "current_proposal": current_proposal,
            "user_feedback": user_feedback,
            "max_tokens": max_tokens,
        }
        probe = CoachContext.model_construct(**payload)
        if probe.data_tokens() <= max_tokens:
            return CoachContext.model_validate(payload)
        if activities and _activity_droppable(activities, today):
            droppable = [
                index for index, activity in enumerate(activities) if activity.id not in keep_ids
            ]
            if droppable:
                oldest_index = min(droppable, key=lambda index: activities[index].start_date_local)
                activities.pop(oldest_index)
                continue
        if rollup:
            rollup.pop(0)
        elif wellness_rows:
            wellness_rows.pop()
        elif past_events:
            oldest = min(
                range(len(past_events)), key=lambda index: past_events[index].start_date_local
            )
            past_events.pop(oldest)
        elif events:
            furthest = max(range(len(events)), key=lambda index: events[index].start_date_local)
            events.pop(furthest)
        elif activity_detail is not None:
            activity_detail = None
        elif splits:
            splits.pop()
        elif current_proposal is not None:
            current_proposal = None
        elif user_feedback is not None:
            user_feedback = None
        elif races:
            races.pop()
        elif activities:
            candidates = [
                index for index, activity in enumerate(activities) if activity.id not in keep_ids
            ]
            index = min(
                candidates or range(len(activities)),
                key=lambda candidate: activities[candidate].start_date_local,
            )
            activities.pop(index)
        else:
            raise RuntimeError(f"cannot fit the context data in token budget: {max_tokens}")

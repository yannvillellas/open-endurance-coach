from datetime import date, timedelta

from open_endurance_coach.errors import InternalError
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
    keep_ids = activity_keep_ids or set()
    ctx = CoachContext.model_construct(
        focus=focus,
        today=today,
        recent_activities=list(recent_activities),
        activity_detail=activity_detail,
        activity_splits=list(activity_splits or []),
        wellness=list(wellness),
        recent_events=list(recent_events or []),
        upcoming_events=list(upcoming_events),
        goal_races=list(goal_races or []),
        training_rollup=list(training_rollup or []),
        sport_settings=list(sport_settings),
        current_proposal=current_proposal,
        user_feedback=user_feedback,
        max_tokens=max_tokens,
    )
    while ctx.data_tokens() > max_tokens:
        activities = ctx.recent_activities
        if activities and _activity_droppable(activities, today):
            droppable = [
                index for index, activity in enumerate(activities) if activity.id not in keep_ids
            ]
            if droppable:
                oldest_index = min(droppable, key=lambda index: activities[index].start_date_local)
                activities.pop(oldest_index)
                continue
        if ctx.training_rollup:
            ctx.training_rollup.pop(0)
        elif ctx.wellness:
            ctx.wellness.pop()
        elif ctx.recent_events:
            oldest = min(
                range(len(ctx.recent_events)),
                key=lambda index: ctx.recent_events[index].start_date_local,
            )
            ctx.recent_events.pop(oldest)
        elif ctx.upcoming_events:
            furthest = max(
                range(len(ctx.upcoming_events)),
                key=lambda index: ctx.upcoming_events[index].start_date_local,
            )
            ctx.upcoming_events.pop(furthest)
        elif ctx.activity_splits:
            ctx.activity_splits.pop()
        elif ctx.activity_detail is not None:
            ctx.activity_detail = None
        elif ctx.current_proposal is not None:
            ctx.current_proposal = None
        elif ctx.user_feedback is not None:
            ctx.user_feedback = None
        elif ctx.goal_races:
            ctx.goal_races.pop()
        elif ctx.recent_activities:
            candidates = [
                index
                for index, activity in enumerate(ctx.recent_activities)
                if activity.id not in keep_ids
            ]
            index = min(
                candidates or range(len(ctx.recent_activities)),
                key=lambda candidate: ctx.recent_activities[candidate].start_date_local,
            )
            ctx.recent_activities.pop(index)
        else:
            raise InternalError(f"cannot fit the context data in token budget: {max_tokens}")
    return CoachContext.model_validate(ctx.model_dump(mode="json"))

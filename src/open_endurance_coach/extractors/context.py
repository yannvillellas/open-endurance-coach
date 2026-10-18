"""One context gather for every extraction path, driven by a declared plan."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any, cast

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.extractors.periodization import derive_block_plan
from open_endurance_coach.extractors.splits import per_km_splits, stream_types
from open_endurance_coach.extractors.standard import (
    ACTIVITY_LOOKBACK_DAYS,
    DEFAULT_MAX_TOKENS,
    PLAN_HORIZON_DAYS,
    RECENT_EVENT_DAYS,
    UPCOMING_DAYS,
    WELLNESS_LOOKBACK_DAYS,
    fetch_goal_races,
    fetch_training_rollup,
)
from open_endurance_coach.schemas.context import CoachContext, GoalRace, TrainingWeek
from open_endurance_coach.schemas.intervals import (
    Activity,
    ActivitySplit,
    Event,
    SportSettings,
    Wellness,
)

logger = logging.getLogger(__name__)

REFERENCE_WINDOW_DAYS = 3
SPLIT_COVERAGE_TOLERANCE = 0.02

RIDE_TYPES = frozenset(
    {
        "Ride",
        "VirtualRide",
        "GravelRide",
        "MountainBikeRide",
        "EBikeRide",
        "TrackRide",
        "Cyclocross",
        "Velomobile",
    }
)


@dataclass(frozen=True)
class DateRange:
    """Inclusive date window."""

    start: date
    end: date

    def clamped(self, max_days: int) -> "DateRange":
        """Keep at most max_days between start and end (inclusive window: max_days + 1 dates)."""
        if (self.end - self.start).days <= max_days:
            return self
        return DateRange(start=self.end - timedelta(days=max_days), end=self.end)

    def union(self, other: "DateRange") -> "DateRange":
        return DateRange(start=min(self.start, other.start), end=max(self.end, other.end))


def _warn_on_split_coverage(
    activity_id: str, declared_m: float | None, splits: list[ActivitySplit]
) -> None:
    """Note when the split table does not cover the activity's own distance."""
    if declared_m is None or declared_m <= 0:
        return
    covered = sum(split.distance_m for split in splits)
    if not splits or abs(covered - declared_m) / declared_m > SPLIT_COVERAGE_TOLERANCE:
        logger.warning("splits for %r cover %.0f of %.0f m", activity_id, covered, declared_m)


@dataclass(frozen=True)
class ContextPlan:
    """Windows and activity selection for one gather."""

    activities: tuple[DateRange, ...]
    wellness: tuple[DateRange, ...]
    events: tuple[DateRange, ...]
    races: DateRange
    rollup: DateRange
    activity_types: frozenset[str] = frozenset()
    metric_focus: str | None = None
    reference: date | None = None
    detail: bool = False
    illness: bool = False
    elite: bool = False

    @classmethod
    def standard(cls, today: date) -> "ContextPlan":
        return cls(
            activities=(
                DateRange(
                    today - timedelta(days=ACTIVITY_LOOKBACK_DAYS), today + timedelta(days=1)
                ),
            ),
            wellness=(
                DateRange(
                    today - timedelta(days=WELLNESS_LOOKBACK_DAYS), today + timedelta(days=1)
                ),
            ),
            events=(
                DateRange(
                    today - timedelta(days=RECENT_EVENT_DAYS), today + timedelta(days=UPCOMING_DAYS)
                ),
            ),
            races=DateRange(today, today + timedelta(days=PLAN_HORIZON_DAYS)),
            rollup=DateRange(today - timedelta(days=PLAN_HORIZON_DAYS), today),
        )

    def for_activities(
        self,
        *,
        windows: tuple[DateRange, ...],
        activity_types: frozenset[str] = frozenset(),
        metric_focus: str | None = None,
        reference: date | None = None,
        detail: bool = False,
    ) -> "ContextPlan":
        return replace(
            self,
            activities=windows,
            activity_types=activity_types,
            metric_focus=metric_focus,
            reference=reference,
            detail=detail,
        )

    def with_windows(
        self,
        *,
        wellness: tuple[DateRange, ...] | None = None,
        events: tuple[DateRange, ...] | None = None,
    ) -> "ContextPlan":
        return replace(
            self,
            wellness=wellness if wellness is not None else self.wellness,
            events=events if events is not None else self.events,
        )


def _relevance(plan: ContextPlan) -> Callable[[Activity], Any]:
    if plan.metric_focus == "elevation":
        return lambda activity: activity.total_elevation_gain or 0.0
    if plan.metric_focus == "heart_rate":
        return lambda activity: activity.average_heartrate or 0.0
    return lambda activity: activity.start_date_local


def _sport_rank(activity: Activity, plan: ContextPlan) -> bool:
    """The named sport ranks first; no activity is ever dropped by its type."""
    return not plan.activity_types or activity.type in plan.activity_types


async def _load_detail(
    client: IntervalsReadClient, detail_source: Activity
) -> tuple[Activity | None, list[ActivitySplit]]:
    try:
        detail_raw = await client.get_activity(detail_source.id, intervals=True)
        detail = Activity.model_validate(detail_raw)
    except (IntervalsApiError, ValueError) as exc:
        logger.warning("could not read the detail of %r: %r", detail_source.id, exc)
        return None, []
    except Exception:
        logger.exception("unexpected error reading the detail of %r", detail_source.id)
        return None, []
    try:
        speed_based = detail.type in RIDE_TYPES
        streams = await client.get_activity_streams(detail_source.id, stream_types(speed_based))
        splits = per_km_splits(streams, speed_based=speed_based)
        _warn_on_split_coverage(detail_source.id, detail.distance, splits)
        return detail, splits
    except (IntervalsApiError, ValueError) as exc:
        logger.warning(
            "could not read the streams of %r; splits skipped: %r", detail_source.id, exc
        )
    except Exception:
        logger.exception(
            "unexpected error reading the streams of %r; splits skipped", detail_source.id
        )
    return detail, []


def _merge_wellness(batches: list[list[dict[str, Any]]]) -> list[Wellness]:
    seen: set[date] = set()
    merged: list[Wellness] = []
    for batch in batches:
        for item in batch:
            row = Wellness.model_validate(item)
            if row.id not in seen:
                seen.add(row.id)
                merged.append(row)
    return merged


def _merge_events(batches: list[list[dict[str, Any]]]) -> list[Event]:
    seen: set[tuple[object, ...]] = set()
    merged: list[Event] = []
    for batch in batches:
        for item in batch:
            event = Event.model_validate(item)
            key = (
                ("id", event.id)
                if event.id is not None
                else ("fields", event.name, event.start_date_local.isoformat(), event.category)
            )
            if key not in seen:
                seen.add(key)
                merged.append(event)
    return merged


async def gather_context(
    client: IntervalsReadClient,
    plan: ContextPlan,
    focus: str,
    *,
    user_feedback: str | None = None,
    max_tokens: int | None = None,
    today: date,
    review_execution: bool = False,
    review_default: bool = False,
) -> CoachContext:
    async def fetch_activities() -> list[Activity]:
        batches = await asyncio.gather(
            *(
                client.list_activities(window.start.isoformat(), window.end.isoformat())
                for window in plan.activities
            )
        )
        seen: set[str] = set()
        merged: list[Activity] = []
        for batch in batches:
            for item in batch:
                activity = Activity.model_validate(item)
                if activity.id not in seen:
                    seen.add(activity.id)
                    merged.append(activity)
        return merged

    activities = await fetch_activities()
    keep_ids: set[str] = set()
    if plan.detail and plan.reference is not None:
        reference = plan.reference
        window = timedelta(days=REFERENCE_WINDOW_DAYS)
        keep_ids = {
            activity.id
            for activity in activities
            if reference - window <= activity.start_date_local.date() <= reference + window
        }
    activities = sorted(
        activities,
        key=lambda activity: (_sport_rank(activity, plan), _relevance(plan)(activity)),
        reverse=True,
    )
    detail_source: Activity | None = None
    if plan.detail:
        if plan.reference is not None:
            # The cited date wins, then the named sport, then the nearest.
            reference = plan.reference
            candidates = [activity for activity in activities if activity.id in keep_ids]
            candidates.sort(
                key=lambda activity: (
                    0 if _sport_rank(activity, plan) else 1,
                    abs((activity.start_date_local.date() - reference).days),
                )
            )
            detail_source = candidates[0] if candidates else None
        elif activities:
            detail_source = activities[0]

    async def load_detail() -> tuple[Activity | None, list[ActivitySplit]]:
        if detail_source is None:
            return None, []
        return await _load_detail(client, detail_source)

    wellness_calls = [
        client.list_wellness(window.start.isoformat(), window.end.isoformat())
        for window in plan.wellness
    ]
    event_calls = [
        client.list_events(window.start.isoformat(), window.end.isoformat())
        for window in plan.events
    ]
    results = await asyncio.gather(
        load_detail(),
        *wellness_calls,
        *event_calls,
        fetch_goal_races(client, today, start=plan.races.start, end=plan.races.end),
        fetch_training_rollup(client, today, start=plan.rollup.start, end=plan.rollup.end),
        client.get_sport_settings(),
    )
    detail_result = cast("tuple[Activity | None, list[ActivitySplit]]", results[0])
    wellness_batches = cast("list[list[dict[str, Any]]]", results[1 : 1 + len(wellness_calls)])
    event_batches = cast(
        "list[list[dict[str, Any]]]",
        results[1 + len(wellness_calls) : 1 + len(wellness_calls) + len(event_calls)],
    )
    goal_races = cast("list[GoalRace]", results[-3])
    rollup = cast("list[TrainingWeek]", results[-2])
    settings_raw = cast("list[dict[str, Any]]", results[-1])
    wellness = _merge_wellness(wellness_batches)
    events = _merge_events(event_batches)
    activity_detail, activity_splits = detail_result
    events = sorted(events, key=lambda event: event.start_date_local)
    block_plan = derive_block_plan(
        goal_races, rollup, today=today, illness=plan.illness, elite=plan.elite
    )
    execution_review = None
    if review_execution and block_plan:
        # Local import: analysis imports this module (RIDE_TYPES), so a module-level
        # import would be circular. The digest is computed on the full fetched data,
        # before the budget eviction, so old weeks never depend on what survives it.
        from open_endurance_coach.analysis.execution import (
            aggregate_weeks,
            build_review,
            compare_trajectory,
            match_execution,
        )

        block_start = block_plan[0].week_start if review_default else None
        reviewed_events = [
            event
            for event in events
            if block_start is None or event.start_date_local.date() >= block_start
        ]
        reviewed_activities = [
            activity
            for activity in activities
            if block_start is None or activity.start_date_local.date() >= block_start
        ]
        match = match_execution(reviewed_events, reviewed_activities, today=today)
        weeks = aggregate_weeks(match, today=today, origin=block_plan[0].week_start)
        efficacy = compare_trajectory(weeks, block_plan, rollup)
        execution_review = build_review(match, weeks, efficacy)
    return build_within_budget(
        focus=focus,
        recent_activities=activities,
        wellness=sorted(wellness, key=lambda row: row.id, reverse=True),
        upcoming_events=[event for event in events if event.start_date_local.date() >= today],
        recent_events=[
            event for event in reversed(events) if event.start_date_local.date() < today
        ],
        goal_races=goal_races,
        block_plan=block_plan,
        execution_review=execution_review,
        training_rollup=rollup,
        sport_settings=[SportSettings.model_validate(item) for item in settings_raw],
        user_feedback=user_feedback,
        activity_keep_ids=keep_ids,
        activity_detail=activity_detail,
        activity_splits=activity_splits,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        today=today,
    )

"""One context gather for every extraction path, driven by a declared plan."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any

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
from open_endurance_coach.schemas.context import CoachContext
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

    activities: DateRange
    wellness: DateRange
    events: DateRange
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
            activities=DateRange(
                today - timedelta(days=ACTIVITY_LOOKBACK_DAYS), today + timedelta(days=1)
            ),
            wellness=DateRange(
                today - timedelta(days=WELLNESS_LOOKBACK_DAYS), today + timedelta(days=1)
            ),
            events=DateRange(
                today - timedelta(days=RECENT_EVENT_DAYS),
                today + timedelta(days=UPCOMING_DAYS),
            ),
            races=DateRange(today, today + timedelta(days=PLAN_HORIZON_DAYS)),
            rollup=DateRange(today - timedelta(days=PLAN_HORIZON_DAYS), today),
        )

    def for_activities(
        self,
        *,
        window: DateRange,
        activity_types: frozenset[str] = frozenset(),
        metric_focus: str | None = None,
        reference: date | None = None,
        detail: bool = False,
    ) -> "ContextPlan":
        return replace(
            self,
            activities=window,
            activity_types=activity_types,
            metric_focus=metric_focus,
            reference=reference,
            detail=detail,
        )

    def with_windows(
        self, *, wellness: DateRange | None = None, events: DateRange | None = None
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


async def gather_context(
    client: IntervalsReadClient,
    plan: ContextPlan,
    focus: str,
    *,
    user_feedback: str | None = None,
    max_tokens: int | None = None,
    today: date,
) -> CoachContext:
    activities_raw = await client.list_activities(
        plan.activities.start.isoformat(), plan.activities.end.isoformat()
    )
    activities = [Activity.model_validate(item) for item in activities_raw]
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

    (
        detail_result,
        wellness_raw,
        events_raw,
        goal_races,
        rollup,
        settings_raw,
    ) = await asyncio.gather(
        load_detail(),
        client.list_wellness(plan.wellness.start.isoformat(), plan.wellness.end.isoformat()),
        client.list_events(plan.events.start.isoformat(), plan.events.end.isoformat()),
        fetch_goal_races(client, today, start=plan.races.start, end=plan.races.end),
        fetch_training_rollup(client, today, start=plan.rollup.start, end=plan.rollup.end),
        client.get_sport_settings(),
    )
    activity_detail, activity_splits = detail_result
    events = sorted(
        (Event.model_validate(item) for item in events_raw),
        key=lambda event: event.start_date_local,
    )
    return build_within_budget(
        focus=focus,
        recent_activities=activities,
        wellness=sorted(
            (Wellness.model_validate(item) for item in wellness_raw),
            key=lambda row: row.id,
            reverse=True,
        ),
        upcoming_events=[event for event in events if event.start_date_local.date() >= today],
        recent_events=[
            event for event in reversed(events) if event.start_date_local.date() < today
        ],
        goal_races=goal_races,
        block_plan=derive_block_plan(
            goal_races, rollup, today=today, illness=plan.illness, elite=plan.elite
        ),
        training_rollup=rollup,
        sport_settings=[SportSettings.model_validate(item) for item in settings_raw],
        user_feedback=user_feedback,
        activity_keep_ids=keep_ids,
        activity_detail=activity_detail,
        activity_splits=activity_splits,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        today=today,
    )

"""One context gather for every extraction path, driven by a declared plan."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from open_endurance_coach.clients.intervals import IntervalsApiError
from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.extractors.splits import per_km_splits, stream_types
from open_endurance_coach.extractors.standard import (
    ACTIVITY_LOOKBACK_DAYS,
    DEFAULT_MAX_TOKENS,
    RACE_HORIZON_DAYS,
    RECENT_EVENT_DAYS,
    ROLLUP_LOOKBACK_DAYS,
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

    activities_days: int
    wellness_days: int
    event_past_days: int
    event_future_days: int
    race_horizon_days: int
    rollup_days: int
    activity_types: frozenset[str] = frozenset()
    metric_focus: str | None = None
    reference: date | None = None
    detail: bool = False

    @classmethod
    def standard(cls) -> "ContextPlan":
        return cls(
            activities_days=ACTIVITY_LOOKBACK_DAYS,
            wellness_days=WELLNESS_LOOKBACK_DAYS,
            event_past_days=RECENT_EVENT_DAYS,
            event_future_days=UPCOMING_DAYS,
            race_horizon_days=RACE_HORIZON_DAYS,
            rollup_days=ROLLUP_LOOKBACK_DAYS,
        )

    def for_activities(
        self,
        *,
        days: int,
        activity_types: frozenset[str] = frozenset(),
        metric_focus: str | None = None,
        reference: date | None = None,
        detail: bool = False,
    ) -> "ContextPlan":
        return replace(
            self,
            activities_days=days,
            activity_types=activity_types,
            metric_focus=metric_focus,
            reference=reference,
            detail=detail,
        )


def _relevance(plan: ContextPlan) -> Callable[[Activity], Any]:
    if plan.metric_focus == "elevation":
        return lambda activity: activity.total_elevation_gain or 0.0
    if plan.metric_focus == "heart_rate":
        return lambda activity: activity.average_heartrate or 0.0
    return lambda activity: activity.start_date_local


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
    newest = (today + timedelta(days=1)).isoformat()
    activities_raw = await client.list_activities(
        (today - timedelta(days=plan.activities_days)).isoformat(), newest
    )
    activities = [Activity.model_validate(item) for item in activities_raw]
    if plan.activity_types:
        activities = [activity for activity in activities if activity.type in plan.activity_types]
    keep_ids: set[str] = set()
    if plan.detail and plan.reference is not None:
        reference = plan.reference
        window = timedelta(days=REFERENCE_WINDOW_DAYS)
        keep_ids = {
            activity.id
            for activity in activities
            if reference - window <= activity.start_date_local.date() <= reference + window
        }
    activities = sorted(activities, key=_relevance(plan), reverse=True)
    detail_source: Activity | None = None
    if plan.detail:
        if plan.reference is not None:
            # The cited date wins: exact day first, then the nearest.
            reference = plan.reference
            candidates = [activity for activity in activities if activity.id in keep_ids]
            candidates.sort(
                key=lambda activity: abs((activity.start_date_local.date() - reference).days)
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
        client.list_wellness((today - timedelta(days=plan.wellness_days)).isoformat(), newest),
        client.list_events(
            (today - timedelta(days=plan.event_past_days)).isoformat(),
            (today + timedelta(days=plan.event_future_days)).isoformat(),
        ),
        fetch_goal_races(client, today, horizon_days=plan.race_horizon_days),
        fetch_training_rollup(client, today, lookback_days=plan.rollup_days),
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
        training_rollup=rollup,
        sport_settings=[SportSettings.model_validate(item) for item in settings_raw],
        user_feedback=user_feedback,
        activity_keep_ids=keep_ids,
        activity_detail=activity_detail,
        activity_splits=activity_splits,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        today=today,
    )


class StandardExtractor:
    def __init__(self, settings: Settings, client: IntervalsReadClient) -> None:
        self._settings = settings
        self._client = client

    def _today(self, today: date | None) -> date:
        return today or datetime.now(ZoneInfo(self._settings.app_timezone)).date()

    async def extract(
        self,
        focus: str,
        *,
        user_feedback: str | None = None,
        max_tokens: int | None = None,
        today: date | None = None,
    ) -> CoachContext:
        return await gather_context(
            self._client,
            ContextPlan.standard(),
            focus,
            user_feedback=user_feedback,
            max_tokens=max_tokens,
            today=self._today(today),
        )

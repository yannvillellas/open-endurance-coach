import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.extractors.standard import (
    DEFAULT_MAX_TOKENS,
    UPCOMING_DAYS,
    WELLNESS_LOOKBACK_DAYS,
)
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.intervals import Activity, Event, SportSettings, Wellness

DEFAULT_DEEP_LOOKBACK_DAYS = 90
RIDE_TYPES = frozenset(
    {"Ride", "VirtualRide", "GravelRide", "TrackRide", "Cyclocross", "MountainBikeRide"}
)

_TREND_RE = re.compile(r"\b(trend|improve|progress|evolution)\b", re.IGNORECASE)
_DURATION_RE = re.compile(r"last (\d+) (day|week|month)s?", re.IGNORECASE)
_HILL_RE = re.compile(r"\b(hills?|hilly|climbs?|elevation)\b", re.IGNORECASE)
_HEART_RATE_RE = re.compile(r"\b(heart ?rate|hr)\b", re.IGNORECASE)


@dataclass(frozen=True)
class DeepQuery:
    lookback_days: int
    metric_focus: str | None = None
    activity_types: frozenset[str] = frozenset()


def detect_deep_query(focus: str) -> DeepQuery | None:
    if not _TREND_RE.search(focus):
        return None
    lookback = DEFAULT_DEEP_LOOKBACK_DAYS
    duration = _DURATION_RE.search(focus)
    if duration:
        amount = int(duration.group(1))
        unit = duration.group(2).lower()
        lookback = amount * (7 if unit == "week" else 30 if unit == "month" else 1)
    metric = (
        "elevation"
        if _HILL_RE.search(focus)
        else "heart_rate"
        if _HEART_RATE_RE.search(focus)
        else None
    )
    activity_types = frozenset({"Ride"}) if _HILL_RE.search(focus) else frozenset()
    return DeepQuery(lookback_days=lookback, metric_focus=metric, activity_types=activity_types)


class DeepHistoricalExtractor:
    def __init__(self, settings: Settings, client: IntervalsReadClient) -> None:
        self._settings = settings
        self._client = client

    def _today(self, today: date | None) -> date:
        return today or datetime.now(ZoneInfo(self._settings.app_timezone)).date()

    def _relevance(self, query: DeepQuery) -> Callable[[Activity], Any]:
        if query.metric_focus == "elevation":
            return lambda activity: activity.total_elevation_gain or 0.0
        if query.metric_focus == "heart_rate":
            return lambda activity: activity.average_heartrate or 0.0
        return lambda activity: activity.start_date_local

    async def extract(
        self,
        focus: str,
        *,
        user_feedback: str | None = None,
        max_tokens: int | None = None,
        today: date | None = None,
    ) -> CoachContext:
        query = detect_deep_query(focus)
        if query is None:
            raise ValueError(f"focus is not a deep query: {focus!r}")
        current = self._today(today)
        newest = (current + timedelta(days=1)).isoformat()
        activities_raw = await self._client.list_activities(
            (current - timedelta(days=query.lookback_days)).isoformat(), newest
        )
        activities = [Activity.model_validate(item) for item in activities_raw]
        if query.activity_types:
            activities = [a for a in activities if a.type in query.activity_types]
        activities = sorted(activities, key=self._relevance(query), reverse=True)
        activity_detail = None
        if activities:
            detail_raw = await self._client.get_activity(activities[0].id, intervals=True)
            activity_detail = Activity.model_validate(detail_raw)
        wellness_raw = await self._client.list_wellness(
            (current - timedelta(days=WELLNESS_LOOKBACK_DAYS)).isoformat(), newest
        )
        events_raw = await self._client.list_events(
            current.isoformat(), (current + timedelta(days=UPCOMING_DAYS)).isoformat()
        )
        settings_raw = await self._client.get_sport_settings()
        return build_within_budget(
            focus=focus,
            recent_activities=activities,
            wellness=sorted(
                (Wellness.model_validate(item) for item in wellness_raw),
                key=lambda row: row.id,
                reverse=True,
            ),
            upcoming_events=sorted(
                (Event.model_validate(item) for item in events_raw),
                key=lambda event: event.start_date_local,
            ),
            sport_settings=[SportSettings.model_validate(item) for item in settings_raw],
            user_feedback=user_feedback,
            activity_detail=activity_detail,
            max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        )

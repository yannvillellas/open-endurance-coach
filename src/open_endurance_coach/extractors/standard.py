import logging
from datetime import date, datetime, timedelta
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.schemas.context import (
    CoachContext,
    GoalRace,
    MacroPhase,
    SportWeek,
    TrainingWeek,
)
from open_endurance_coach.schemas.decisions import RACE_CATEGORIES
from open_endurance_coach.schemas.intervals import Activity, Event, SportSettings, Wellness

ACTIVITY_LOOKBACK_DAYS = 14
WELLNESS_LOOKBACK_DAYS = 7
RECENT_EVENT_DAYS = 14
UPCOMING_DAYS = 14
RACE_HORIZON_DAYS = 120
ROLLUP_LOOKBACK_DAYS = 90
DEFAULT_MAX_TOKENS = 8192
logger = logging.getLogger(__name__)
RACE_CATEGORY_FILTER = ",".join(RACE_CATEGORIES)


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def training_rollup(summary_rows: list[dict[str, Any]], *, today: date) -> list[TrainingWeek]:
    parsed: dict[date, dict[str, Any]] = {}
    for row in summary_rows:
        raw = row.get("date")
        if not isinstance(raw, str):
            raise ValueError("athlete-summary row without a date")
        parsed[date.fromisoformat(raw)] = row
    if not parsed:
        return []
    starts = sorted(parsed)
    gaps = {(later - earlier).days for earlier, later in pairwise(starts)}
    if any(gap <= 0 or gap % 7 for gap in gaps):
        raise ValueError(f"unexpected athlete-summary bucket spacing: {sorted(gaps)}")
    weeks: list[TrainingWeek] = []
    cursor = starts[0]
    while cursor <= starts[-1]:
        row = parsed.get(cursor, {})
        sports = [
            SportWeek(
                category=str(item.get("category") or "Unknown"),
                sessions=int(item.get("count") or 0),
                time_s=int(item.get("time") or 0),
                load=_float_or_none(item.get("training_load")),
            )
            for item in row.get("byCategory") or []
            if int(item.get("count") or 0) > 0
        ]
        weeks.append(
            TrainingWeek(
                week_start=cursor,
                partial=cursor <= today < cursor + timedelta(days=7),
                sessions=int(row.get("count") or 0),
                time_s=int(row.get("time") or 0),
                load=_float_or_none(row.get("training_load")),
                fitness=_float_or_none(row.get("fitness")),
                fatigue=_float_or_none(row.get("fatigue")),
                form=_float_or_none(row.get("form")),
                ramp_rate=_float_or_none(row.get("rampRate")),
                sports=sports,
            )
        )
        cursor += timedelta(days=7)
    return weeks


def macro_phase(days_to_race: int) -> MacroPhase:
    if days_to_race <= 0:
        return "Race week"
    if days_to_race <= 7:
        return "Taper"
    if days_to_race <= 28:
        return "Peak"
    if days_to_race <= 84:
        return "Build"
    return "Base"


def goal_race(event: Event, *, today: date) -> GoalRace | None:
    if event.category not in RACE_CATEGORIES:
        return None
    race_date = event.start_date_local.date()
    days = (race_date - today).days
    if days < 0:
        return None
    return GoalRace(
        event_id=event.id,
        name=event.name,
        date=race_date,
        category=event.category,
        type=event.type,
        days_to_race=days,
        weeks_to_race=(days + 6) // 7,
        phase=macro_phase(days),
        moving_time=event.moving_time,
        distance=event.distance,
        icu_training_load=event.icu_training_load,
    )


async def fetch_goal_races(client: IntervalsReadClient, current: date) -> list[GoalRace]:
    rows = await client.list_events(
        current.isoformat(),
        (current + timedelta(days=RACE_HORIZON_DAYS)).isoformat(),
        category=RACE_CATEGORY_FILTER,
    )
    return sorted(
        (
            race
            for race in (goal_race(Event.model_validate(item), today=current) for item in rows)
            if race is not None
        ),
        key=lambda race: race.date,
    )


async def fetch_training_rollup(client: IntervalsReadClient, current: date) -> list[TrainingWeek]:
    rows = await client.get_athlete_summary(
        start=(current - timedelta(days=ROLLUP_LOOKBACK_DAYS)).isoformat(),
        end=current.isoformat(),
    )
    try:
        return training_rollup(rows, today=current)
    except ValueError as exc:
        logger.warning("dropping the training rollup: %s", exc)
        return []


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
        current = self._today(today)
        newest = (current + timedelta(days=1)).isoformat()
        activities_raw = await self._client.list_activities(
            (current - timedelta(days=ACTIVITY_LOOKBACK_DAYS)).isoformat(), newest
        )
        wellness_raw = await self._client.list_wellness(
            (current - timedelta(days=WELLNESS_LOOKBACK_DAYS)).isoformat(), newest
        )
        events_raw = await self._client.list_events(
            (current - timedelta(days=RECENT_EVENT_DAYS)).isoformat(),
            (current + timedelta(days=UPCOMING_DAYS)).isoformat(),
        )
        goal_races = await fetch_goal_races(self._client, current)
        rollup = await fetch_training_rollup(self._client, current)
        settings_raw = await self._client.get_sport_settings()
        activities = sorted(
            (Activity.model_validate(item) for item in activities_raw),
            key=lambda activity: activity.start_date_local,
            reverse=True,
        )
        wellness = sorted(
            (Wellness.model_validate(item) for item in wellness_raw),
            key=lambda row: row.id,
            reverse=True,
        )
        events = sorted(
            (Event.model_validate(item) for item in events_raw),
            key=lambda event: event.start_date_local,
        )
        recent_events = [
            event for event in reversed(events) if event.start_date_local.date() < current
        ]
        upcoming_events = [event for event in events if event.start_date_local.date() >= current]
        return build_within_budget(
            focus=focus,
            recent_activities=activities,
            wellness=wellness,
            upcoming_events=upcoming_events,
            recent_events=recent_events,
            goal_races=goal_races,
            training_rollup=rollup,
            sport_settings=[SportSettings.model_validate(item) for item in settings_raw],
            user_feedback=user_feedback,
            activity_detail=None,
            max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
            today=current,
        )

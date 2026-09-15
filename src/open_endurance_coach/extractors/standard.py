from datetime import date, datetime, timedelta
from typing import get_args
from zoneinfo import ZoneInfo

from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.schemas.context import CoachContext, GoalRace, MacroPhase
from open_endurance_coach.schemas.decisions import RaceCategory
from open_endurance_coach.schemas.intervals import Activity, Event, SportSettings, Wellness

ACTIVITY_LOOKBACK_DAYS = 14
WELLNESS_LOOKBACK_DAYS = 7
UPCOMING_DAYS = 14
RACE_HORIZON_DAYS = 120
DEFAULT_MAX_TOKENS = 4096
RACE_CATEGORIES: tuple[RaceCategory, ...] = get_args(RaceCategory)
RACE_CATEGORY_FILTER = ",".join(RACE_CATEGORIES)


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
        current = self._today(today)
        newest = (current + timedelta(days=1)).isoformat()
        activities_raw = await self._client.list_activities(
            (current - timedelta(days=ACTIVITY_LOOKBACK_DAYS)).isoformat(), newest
        )
        wellness_raw = await self._client.list_wellness(
            (current - timedelta(days=WELLNESS_LOOKBACK_DAYS)).isoformat(), newest
        )
        events_raw = await self._client.list_events(
            current.isoformat(), (current + timedelta(days=UPCOMING_DAYS)).isoformat()
        )
        races_raw = await self._client.list_events(
            current.isoformat(),
            (current + timedelta(days=RACE_HORIZON_DAYS)).isoformat(),
            category=RACE_CATEGORY_FILTER,
        )
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
        goal_races = sorted(
            (
                race
                for race in (
                    goal_race(Event.model_validate(item), today=current) for item in races_raw
                )
                if race is not None
            ),
            key=lambda race: race.date,
        )
        return build_within_budget(
            focus=focus,
            recent_activities=activities,
            wellness=wellness,
            upcoming_events=events,
            goal_races=goal_races,
            sport_settings=[SportSettings.model_validate(item) for item in settings_raw],
            user_feedback=user_feedback,
            activity_detail=None,
            max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
            today=current,
        )

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.intervals import Activity, Event, SportSettings, Wellness

ACTIVITY_LOOKBACK_DAYS = 14
WELLNESS_LOOKBACK_DAYS = 7
UPCOMING_DAYS = 14
DEFAULT_MAX_TOKENS = 4096


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
        return build_within_budget(
            focus=focus,
            recent_activities=activities,
            wellness=wellness,
            upcoming_events=events,
            sport_settings=[SportSettings.model_validate(item) for item in settings_raw],
            user_feedback=user_feedback,
            activity_detail=None,
            max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        )

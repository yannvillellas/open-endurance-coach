import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.context import RIDE_TYPES, ContextPlan, gather_context
from open_endurance_coach.extractors.standard import ACTIVITY_LOOKBACK_DAYS
from open_endurance_coach.schemas.context import CoachContext

DEFAULT_DEEP_LOOKBACK_DAYS = 90
MAX_DEEP_LOOKBACK_DAYS = 3650

_TREND_RE = re.compile(r"\b(trend|improve|progress|evolution)\b", re.IGNORECASE)
_DURATION_RE = re.compile(r"last (\d+) (day|week|month)s?", re.IGNORECASE)
_HILL_RE = re.compile(r"\b(hills?|hilly|climbs?|elevation)\b", re.IGNORECASE)
_HEART_RATE_RE = re.compile(r"\b(heart ?rate|hr)\b", re.IGNORECASE)
_RIDE_RE = re.compile(
    r"\b(ride|rides|riding|bike|biking|cycle|cycling|zwift|gravel|mtb)\b", re.IGNORECASE
)
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_WORDED_DATE_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\s+(20\d{2})\b", re.IGNORECASE
)
_DAY_MONTH_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\b", re.IGNORECASE)
_PAST_REFERENCE_RE = re.compile(
    r"\b(last year|previous year|a year ago|one year ago|years ago)\b", re.IGNORECASE
)
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _referenced_date(focus: str, today: date) -> date | None:
    iso = _ISO_DATE_RE.search(focus)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
    worded = _WORDED_DATE_RE.search(focus)
    if worded:
        month = _MONTHS.get(worded.group(2).lower())
        if month is not None:
            try:
                return date(int(worded.group(3)), month, int(worded.group(1)))
            except ValueError:
                return None
    if _PAST_REFERENCE_RE.search(focus):
        day_month = _DAY_MONTH_RE.search(focus)
        if day_month is not None:
            month = _MONTHS.get(day_month.group(2).lower())
            if month is not None:
                try:
                    return date(today.year - 1, month, int(day_month.group(1)))
                except ValueError:
                    return None
        return today - timedelta(days=365)
    return None


@dataclass(frozen=True)
class DeepQuery:
    lookback_days: int
    metric_focus: str | None = None
    activity_types: frozenset[str] = frozenset()
    reference: date | None = None


def detect_deep_query(focus: str, *, today: date | None = None) -> DeepQuery | None:
    current = today or date.today()
    reference = _referenced_date(focus, current)
    stale_reference = reference is not None and (current - reference).days >= ACTIVITY_LOOKBACK_DAYS
    if not _TREND_RE.search(focus) and not stale_reference:
        return None
    lookback = DEFAULT_DEEP_LOOKBACK_DAYS
    duration = _DURATION_RE.search(focus)
    if duration:
        amount = int(duration.group(1))
        unit = duration.group(2).lower()
        lookback = amount * (7 if unit == "week" else 30 if unit == "month" else 1)
    lookback = min(lookback, MAX_DEEP_LOOKBACK_DAYS)
    if stale_reference:
        assert reference is not None
        lookback = max(lookback, (current - reference).days + 7)
        lookback = min(lookback, MAX_DEEP_LOOKBACK_DAYS)
    if _HEART_RATE_RE.search(focus):
        metric = "heart_rate"
    elif _HILL_RE.search(focus):
        metric = "elevation"
    else:
        metric = None
    activity_types = RIDE_TYPES if metric is not None and _RIDE_RE.search(focus) else frozenset()
    return DeepQuery(
        lookback_days=lookback,
        metric_focus=metric,
        activity_types=activity_types,
        reference=reference,
    )


class DeepHistoricalExtractor:
    def __init__(self, settings: Settings, client: IntervalsReadClient) -> None:
        self._settings = settings
        self._client = client

    def _today(self, today: date | None) -> date:
        return today or datetime.now(ZoneInfo(self._settings.app_timezone)).date()

    async def extract(
        self,
        focus: str,
        *,
        query: DeepQuery | None,
        user_feedback: str | None = None,
        max_tokens: int | None = None,
        today: date | None = None,
    ) -> CoachContext:
        if query is None:
            raise ValueError(f"focus is not a deep query: {focus!r}")
        plan = ContextPlan.standard().for_activities(
            days=query.lookback_days,
            activity_types=query.activity_types,
            metric_focus=query.metric_focus,
            reference=query.reference,
            detail=True,
        )
        return await gather_context(
            self._client,
            plan,
            focus,
            user_feedback=user_feedback,
            max_tokens=max_tokens,
            today=self._today(today),
        )

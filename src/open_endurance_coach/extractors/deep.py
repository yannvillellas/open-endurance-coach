import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.config import Settings
from open_endurance_coach.extractors.context import (
    REFERENCE_WINDOW_DAYS,
    RIDE_TYPES,
    ContextPlan,
    DateRange,
    gather_context,
)
from open_endurance_coach.extractors.standard import ACTIVITY_LOOKBACK_DAYS
from open_endurance_coach.schemas.context import CoachContext

DEFAULT_DEEP_LOOKBACK_DAYS = 90
MAX_DEEP_LOOKBACK_DAYS = 3650
MAX_WELLNESS_WINDOW_DAYS = 365
MAX_EVENT_WINDOW_DAYS = 365

_TREND_RE = re.compile(
    r"\b(trend|improve|progress|evolution|tendance|évolution|progression|amélioration)\b",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(r"last (\d+) (day|week|month)s?", re.IGNORECASE)
_DURATION_FR_RE = re.compile(
    r"\b(\d+)\s+(?:derniers?|dernières?)\s+(jours?|semaines?|mois)\b", re.IGNORECASE
)
_AGO_RE = re.compile(r"\b(\d+)\s+(days?|weeks?|months?)\s+ago\b", re.IGNORECASE)
_AGO_FR_RE = re.compile(r"\bil y a\s+(\d+)\s+(jours?|semaines?|mois)\b", re.IGNORECASE)
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
    window: DateRange
    focus_window: DateRange | None = None
    metric_focus: str | None = None
    activity_types: frozenset[str] = frozenset()
    reference: date | None = None


def _unit_days(unit: str) -> int:
    lowered = unit.lower()
    if lowered.startswith(("day", "jour")):
        return 1
    if lowered.startswith(("week", "semaine")):
        return 7
    return 30


def _week_window(weeks_ago: int, today: date) -> DateRange:
    monday = today - timedelta(days=today.weekday()) - timedelta(weeks=weeks_ago)
    return DateRange(monday, monday + timedelta(days=6))


def detect_deep_query(focus: str, *, today: date | None = None) -> DeepQuery | None:
    current = today or date.today()
    cited = _referenced_date(focus, current)
    stale_reference = cited is not None and (current - cited).days >= ACTIVITY_LOOKBACK_DAYS
    ago = _AGO_RE.search(focus) or _AGO_FR_RE.search(focus)
    duration = _DURATION_RE.search(focus) or _DURATION_FR_RE.search(focus)
    if not (_TREND_RE.search(focus) or stale_reference or ago or duration):
        return None

    point: date | None = None
    weeks_ago: int | None = None
    if ago is not None:
        amount = int(ago.group(1))
        if _unit_days(ago.group(2)) == 7:
            weeks_ago = amount
        else:
            point = current - timedelta(days=amount * _unit_days(ago.group(2)))
    elif stale_reference:
        point = cited

    focus_window: DateRange | None
    if weeks_ago is not None:
        window = _week_window(weeks_ago, current)
        focus_window = window
    else:
        if duration is not None:
            span = int(duration.group(1)) * _unit_days(duration.group(2))
        elif point is not None:
            span = DEFAULT_DEEP_LOOKBACK_DAYS
        else:
            span = DEFAULT_DEEP_LOOKBACK_DAYS
        if point is not None:
            span = max(span, (current - point).days + 7)
        span = min(span, MAX_DEEP_LOOKBACK_DAYS)
        window = DateRange(current - timedelta(days=span), current + timedelta(days=1))
        if point is not None:
            focus_window = DateRange(
                point - timedelta(days=REFERENCE_WINDOW_DAYS),
                point + timedelta(days=REFERENCE_WINDOW_DAYS),
            )
        elif duration is not None:
            focus_window = window
        else:
            focus_window = None

    if _HEART_RATE_RE.search(focus):
        metric = "heart_rate"
    elif _HILL_RE.search(focus):
        metric = "elevation"
    else:
        metric = None
    activity_types = RIDE_TYPES if metric is not None and _RIDE_RE.search(focus) else frozenset()
    return DeepQuery(
        window=window,
        focus_window=focus_window,
        metric_focus=metric,
        activity_types=activity_types,
        reference=point,
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
        current = self._today(today)
        standard = ContextPlan.standard(current)
        plan = standard.for_activities(
            window=query.window,
            activity_types=query.activity_types,
            metric_focus=query.metric_focus,
            reference=query.reference,
            detail=True,
        )
        if query.focus_window is not None:
            plan = plan.with_windows(
                wellness=query.focus_window.clamped(MAX_WELLNESS_WINDOW_DAYS),
                events=query.focus_window.clamped(MAX_EVENT_WINDOW_DAYS).union(standard.events),
            )
        return await gather_context(
            self._client,
            plan,
            focus,
            user_feedback=user_feedback,
            max_tokens=max_tokens,
            today=current,
        )

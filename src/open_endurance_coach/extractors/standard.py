import logging
from datetime import date, timedelta
from itertools import pairwise
from typing import Any

from open_endurance_coach.clients.protocols import IntervalsReadClient
from open_endurance_coach.schemas.context import GoalRace, SportWeek, TrainingWeek
from open_endurance_coach.schemas.decisions import RACE_CATEGORIES
from open_endurance_coach.schemas.intervals import Event

ACTIVITY_LOOKBACK_DAYS = 14
WELLNESS_LOOKBACK_DAYS = 7
RECENT_EVENT_DAYS = 14
UPCOMING_DAYS = 14
PLAN_HORIZON_DAYS = 224
DEFAULT_MAX_TOKENS = 16384
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
        moving_time=event.moving_time,
        distance=event.distance,
        description=event.description,
        icu_training_load=event.icu_training_load,
    )


async def fetch_goal_races(
    client: IntervalsReadClient,
    current: date,
    *,
    start: date,
    end: date,
) -> list[GoalRace]:
    rows = await client.list_events(
        start.isoformat(),
        end.isoformat(),
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


async def fetch_training_rollup(
    client: IntervalsReadClient,
    current: date,
    *,
    start: date,
    end: date,
) -> list[TrainingWeek]:
    rows = await client.get_athlete_summary(start=start.isoformat(), end=end.isoformat())
    try:
        return training_rollup(rows, today=current)
    except ValueError as exc:
        logger.warning("dropping the training rollup: %r", exc)
        return []

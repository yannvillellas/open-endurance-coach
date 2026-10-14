"""Resolve the analysis window for one question with the model."""

import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

from open_endurance_coach.clients.llm import LlmClient, LlmError, LlmMessage
from open_endurance_coach.extractors.context import RIDE_TYPES, ContextPlan, DateRange

logger = logging.getLogger(__name__)

MAX_ACTIVITY_WINDOW_DAYS = 3650
MAX_WELLNESS_WINDOW_DAYS = 365
MAX_EVENT_WINDOW_DAYS = 365

_RESOLUTION_PROMPT = (
    "You pick which athlete data an endurance coach must read before answering the "
    "next question. Reply with one json object and nothing else:\n"
    '{"window": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} or null, '
    '"sport": "<activity type or null>", '
    '"metric": "heart_rate" or "elevation" or null, '
    '"detail": true or false}\n'
    "- window: the exact past period the question is about; use null when the "
    "question only needs the recent context.\n"
    "- sport: set it only when the question names a sport (Ride, Run, Swim, ...).\n"
    "- metric: set it only when the question is about heart rate or elevation.\n"
    "- detail: true only when the question asks to inspect one specific activity "
    "(splits, per kilometre, a cited session)."
)


class ResolvedWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: date
    end: date


class ResolvedContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window: ResolvedWindow | None = None
    sport: str | None = None
    metric: Literal["heart_rate", "elevation"] | None = None
    detail: bool = False


@dataclass(frozen=True)
class PlanResolution:
    plan: ContextPlan
    notices: tuple[str, ...]


def _resolution_messages(focus: str, today: date) -> list[LlmMessage]:
    today_line = f"Today is {today.isoformat()}."
    return [
        LlmMessage(role="system", content=f"{_RESOLUTION_PROMPT}\n{today_line}"),
        LlmMessage(role="user", content=focus),
    ]


def _validate_resolved(payload: object) -> None:
    ResolvedContext.model_validate(payload)


def _sport_types(sport: str | None) -> frozenset[str]:
    if not sport:
        return frozenset()
    value = sport.strip()
    if value.casefold() in {family.casefold() for family in RIDE_TYPES}:
        return RIDE_TYPES
    return frozenset({value})


def _overlaps(first: DateRange, second: DateRange) -> bool:
    return first.start <= second.end and second.start <= first.end


def _announcement(scope: str, resolved: ResolvedContext) -> str:
    parts = [f"Fetch: {scope}"]
    if resolved.sport:
        parts.append(f"sport: {resolved.sport}")
    if resolved.metric:
        parts.append(f"metric: {resolved.metric.replace('_', ' ')}")
    if resolved.detail:
        parts.append("detail: yes")
    return " · ".join(parts)


def _plan_from_resolution(
    resolved: ResolvedContext, today: date, standard: ContextPlan
) -> PlanResolution:
    if resolved.window is None:
        plan = standard.for_activities(
            window=standard.activities,
            activity_types=_sport_types(resolved.sport),
            metric_focus=resolved.metric,
            reference=None,
            detail=resolved.detail,
        )
        return PlanResolution(plan, (_announcement("standard windows", resolved),))

    start = resolved.window.start
    end = min(resolved.window.end, today)
    if end < start:
        return PlanResolution(
            standard,
            (
                "Fetch: standard windows"
                f" (invalid window ignored: {resolved.window.start} → {resolved.window.end}).",
            ),
        )

    scope = DateRange(start, end)
    activities = scope.clamped(MAX_ACTIVITY_WINDOW_DAYS)
    wellness = scope.clamped(MAX_WELLNESS_WINDOW_DAYS)
    events_scope = scope.clamped(MAX_EVENT_WINDOW_DAYS)
    events = (
        events_scope.union(standard.events)
        if _overlaps(events_scope, standard.events)
        else events_scope
    )
    plan = standard.for_activities(
        window=activities,
        activity_types=_sport_types(resolved.sport),
        metric_focus=resolved.metric,
        reference=end if resolved.detail else None,
        detail=resolved.detail,
    ).with_windows(wellness=wellness, events=events)
    notices = [_announcement(f"{start.isoformat()} → {end.isoformat()}", resolved)]
    if activities != scope:
        notices.append(f"Window clipped: activities keep the last {MAX_ACTIVITY_WINDOW_DAYS} days.")
    if wellness != scope or events_scope != scope:
        notices.append(
            f"Window clipped: wellness and events keep the last {MAX_WELLNESS_WINDOW_DAYS} days."
        )
    return PlanResolution(plan, tuple(notices))


async def resolve_context_plan(
    llm_client: LlmClient,
    focus: str,
    *,
    today: date,
    max_attempts: int = 2,
) -> PlanResolution:
    standard = ContextPlan.standard(today)
    try:
        raw = await llm_client.complete_json(
            _resolution_messages(focus, today),
            max_attempts=max_attempts,
            validator=_validate_resolved,
        )
        resolved = ResolvedContext.model_validate(json.loads(raw))
    except LlmError as exc:
        logger.warning("context window resolution failed: %r; using the standard plan", exc)
        return PlanResolution(standard, ("Fetch: standard windows (resolution failed).",))
    return _plan_from_resolution(resolved, today, standard)

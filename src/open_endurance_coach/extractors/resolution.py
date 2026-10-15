"""Resolve the analysis window for one question with the model."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    "(splits, per kilometre, a cited session); when true, the window end must be that "
    "activity's date.\n"
    "- illness: true only when an active athlete fact says the athlete is currently ill "
    "or sick.\n"
    "- elite: true only when an active athlete fact says the athlete is professional or "
    "elite."
)

_RIDE_TYPES_FOLDED = {family.casefold() for family in RIDE_TYPES}

_SPORT_ALIASES = {
    "bike": "Ride",
    "biking": "Ride",
    "cycling": "Ride",
    "ride": "Ride",
    "run": "Run",
    "running": "Run",
    "jog": "Run",
    "swim": "Swim",
    "swimming": "Swim",
    "walk": "Walk",
    "walking": "Walk",
    "hike": "Hike",
    "hiking": "Hike",
    "row": "Rowing",
    "rowing": "Rowing",
}


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
    illness: bool = False
    elite: bool = False


@dataclass(frozen=True)
class PlanResolution:
    plan: ContextPlan
    notices: tuple[str, ...]


def _resolution_messages(focus: str, today: date, facts: Sequence[str]) -> list[LlmMessage]:
    today_line = f"Today is {today.isoformat()}."
    system = f"{_RESOLUTION_PROMPT}\n{today_line}"
    parts: list[str] = []
    if facts:
        listing = "\n".join(f"- {fact}" for fact in facts)
        parts.append(f"Active athlete facts:\n{listing}")
    parts.append(focus)
    return [
        LlmMessage(role="system", content=system),
        LlmMessage(role="user", content="\n\n".join(parts)),
    ]


def _validate_resolved(payload: object) -> None:
    ResolvedContext.model_validate(payload)


def _sport_types(sport: str | None) -> frozenset[str]:
    if not sport:
        return frozenset()
    folded = sport.strip().casefold()
    canonical = _SPORT_ALIASES.get(folded, sport.strip())
    if canonical.casefold() in _RIDE_TYPES_FOLDED:
        return RIDE_TYPES
    return frozenset({canonical})


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
    states = [
        name
        for name, enabled in (("illness", resolved.illness), ("elite", resolved.elite))
        if enabled
    ]
    if states:
        parts.append(f"state: {', '.join(states)}")
    return " · ".join(parts)


def _with_state(plan: ContextPlan, resolved: ResolvedContext) -> ContextPlan:
    return replace(plan, illness=resolved.illness, elite=resolved.elite)


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
        return PlanResolution(
            _with_state(plan, resolved), (_announcement("standard windows", resolved),)
        )

    start = resolved.window.start
    end = min(resolved.window.end, today)
    if end < start:
        return PlanResolution(
            _with_state(standard, resolved),
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
    plan = _with_state(plan, resolved)
    notices = [_announcement(f"{start.isoformat()} → {end.isoformat()}", resolved)]
    if activities != scope:
        notices.append(
            f"Window clipped: activities keep up to {MAX_ACTIVITY_WINDOW_DAYS + 1} dates."
        )
    if wellness != scope or events_scope != scope:
        notices.append(
            "Window clipped: wellness and past events keep up to "
            f"{MAX_WELLNESS_WINDOW_DAYS + 1} dates."
        )
    return PlanResolution(plan, tuple(notices))


async def resolve_context_plan(
    llm_client: LlmClient,
    focus: str,
    *,
    today: date,
    facts: Sequence[str] = (),
    max_attempts: int = 2,
) -> PlanResolution:
    standard = ContextPlan.standard(today)
    try:
        raw = await llm_client.complete_json(
            _resolution_messages(focus, today, facts),
            max_attempts=max_attempts,
            validator=_validate_resolved,
        )
        resolved = ResolvedContext.model_validate(json.loads(raw))
    except LlmError as exc:
        logger.warning("context window resolution failed: %r; using the standard plan", exc)
        return PlanResolution(standard, ("Fetch: standard windows (resolution failed).",))
    return _plan_from_resolution(resolved, today, standard)

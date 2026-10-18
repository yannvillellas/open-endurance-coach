"""Resolve the analysis window for one question with the model."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict

from open_endurance_coach.chat.history import trim_history
from open_endurance_coach.clients.llm import LlmClient, LlmError, LlmMessage
from open_endurance_coach.extractors.context import RIDE_TYPES, ContextPlan, DateRange
from open_endurance_coach.extractors.standard import PLAN_HORIZON_DAYS
from open_endurance_coach.tokens import estimate_text_tokens

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
    '"detail": true or false, "review_execution": true or false}\n'
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
    "elite.\n"
    "- review_execution: true only when the question asks to compare the past execution "
    "against what was planned (did the athlete follow the plan, how did the weeks go "
    "against it), whatever the wording; false otherwise.\n"
    "A follow-up or clarification keeps the period of the question it refers to."
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
    review_execution: bool = False


@dataclass(frozen=True)
class PlanResolution:
    plan: ContextPlan
    notices: tuple[str, ...]
    review_execution: bool = False


def _messages_tokens(messages: Sequence[LlmMessage]) -> int:
    return sum(estimate_text_tokens(message.content) for message in messages)


def _render_history(history: Sequence[LlmMessage]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in history)


def _resolution_system(today: date) -> str:
    return f"{_RESOLUTION_PROMPT}\nToday is {today.isoformat()}."


def _resolution_context(facts: Sequence[str], focus: str) -> str:
    """The fixed part of the user message, without the conversation."""
    parts: list[str] = []
    if facts:
        listing = "\n".join(f"- {fact}" for fact in facts)
        parts.append(f"Active athlete facts:\n{listing}")
    parts.append(focus)
    return "\n\n".join(parts)


def _resolution_messages(
    focus: str, today: date, facts: Sequence[str], history: Sequence[LlmMessage]
) -> list[LlmMessage]:
    parts: list[str] = []
    transcript = _render_history(history) if history else ""
    if transcript:
        parts.append(f"Conversation so far:\n{transcript}")
    parts.append(_resolution_context(facts, focus))
    return [
        LlmMessage(role="system", content=_resolution_system(today)),
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


def _range_label(window: DateRange) -> str:
    return f"{window.start.isoformat()} → {window.end.isoformat()}"


def _apply_review(plan: ContextPlan, resolved: ResolvedContext, today: date) -> ContextPlan:
    """A review without its own period reads the whole planning horizon."""
    if not resolved.review_execution:
        return plan
    window = DateRange(today - timedelta(days=PLAN_HORIZON_DAYS), today)
    return replace(plan, activities=window, events=window)


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
        plan = _apply_review(_with_state(plan, resolved), resolved, today)
        label = _range_label(plan.activities) if resolved.review_execution else "standard windows"
        return PlanResolution(
            plan,
            (_announcement(label, resolved),),
            review_execution=resolved.review_execution,
        )

    start = resolved.window.start
    end = min(resolved.window.end, today)
    if end < start:
        plan = _apply_review(_with_state(standard, resolved), resolved, today)
        label = _range_label(plan.activities) if resolved.review_execution else "standard windows"
        return PlanResolution(
            plan,
            (
                _announcement(label, resolved)
                + f" · invalid window ignored: {start} → {resolved.window.end}.",
            ),
            review_execution=resolved.review_execution,
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
    return PlanResolution(plan, tuple(notices), review_execution=resolved.review_execution)


async def resolve_context_plan(
    llm_client: LlmClient,
    focus: str,
    *,
    today: date,
    facts: Sequence[str] = (),
    history: Sequence[LlmMessage] = (),
    max_attempts: int = 2,
    max_tokens: int | None = None,
) -> PlanResolution:
    """Resolve the window; with ``max_tokens`` the request is fitted to that budget.

    The fixed parts of the prompt (system, facts, question) are subtracted and the
    conversation is trimmed to the remainder with the same helper as the main call,
    so the request stays within the budget whenever those fixed parts fit; if even
    the trimmed transcript would not fit, the conversation is dropped entirely.
    """
    standard = ContextPlan.standard(today)
    if max_tokens is not None:
        fixed = estimate_text_tokens(_resolution_system(today)) + estimate_text_tokens(
            _resolution_context(facts, focus)
        )
        history = trim_history(list(history), max(1, max_tokens - fixed))
        # The rendered transcript adds a prefix per turn and separators on top of
        # the message contents; the second trim also reserves that envelope.
        envelope = (
            _messages_tokens(_resolution_messages(focus, today, facts, history))
            - fixed
            - _messages_tokens(history)
        )
        history = trim_history(history, max(1, max_tokens - fixed - envelope))
        if _messages_tokens(_resolution_messages(focus, today, facts, history)) > max_tokens:
            history = []
    try:
        raw = await llm_client.complete_json(
            _resolution_messages(focus, today, facts, history),
            max_attempts=max_attempts,
            validator=_validate_resolved,
        )
        resolved = ResolvedContext.model_validate(json.loads(raw))
    except LlmError as exc:
        logger.warning("context window resolution failed: %r; using the standard plan", exc)
        return PlanResolution(standard, ("Fetch: standard windows (resolution failed).",))
    return _plan_from_resolution(resolved, today, standard)

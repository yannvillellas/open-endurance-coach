"""Match the executed activities against the planned calendar, purely.

The plan is the whole calendar: every workout and race event, whoever wrote it.
The hub's pairing leads — an activity whose ``paired_event_id`` points at a
planned event is that session, whatever the dates (a pairing outside the review
is never reused for another session). The remaining sessions take the nearest
unused unpaired activity within one day, by type family.

Sessions dated after yesterday are judged only once an executed activity
matches them: the hub's pairing may settle a future session (done early), and a
session dated today with its activity done is judged — never `missed` today
though, the day is still open, so its floor is `partial` — while unmatched
sessions dated today or later are pending. Executed activities may settle past
sessions; an unmatched activity dated today is not reported as unplanned, the
day is not over. Completion reads the hub's ``compliance`` first, then the
load, duration and distance ratios; a match without comparable numbers counts
as completed. The classes are ``completed`` (>= 80%), ``partial`` (20-80%) and
``missed`` (< 20% or no match); an executed activity with no planned session is
``unplanned``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from open_endurance_coach.extractors.context import RIDE_TYPES
from open_endurance_coach.schemas.context import (
    BlockPhase,
    BlockWeek,
    BlockWeekStatus,
    ExecutionReview,
    ReviewSession,
    ReviewWeek,
    TrainingWeek,
)
from open_endurance_coach.schemas.intervals import Activity, Event

COMPLETED_THRESHOLD = 0.8
PARTIAL_THRESHOLD = 0.2

TYPE_FAMILIES: tuple[frozenset[str], ...] = (
    RIDE_TYPES,
    frozenset({"Run", "TrailRun", "VirtualRun"}),
    frozenset({"Swim"}),
    frozenset({"Walk", "Hike"}),
)

Status = Literal["completed", "partial", "missed"]


@dataclass(frozen=True)
class SessionMatch:
    planned: Event
    activity: Activity | None
    status: Status
    completion: float | None
    paired: bool = False


@dataclass(frozen=True)
class ExecutionMatch:
    sessions: tuple[SessionMatch, ...]
    unplanned: tuple[Activity, ...]
    pending: tuple[Event, ...]


@dataclass(frozen=True)
class WeekReview:
    week_start: date
    planned_load: float | None
    executed_load: float | None
    completed: int
    partial: int
    missed: int
    adherence: float | None
    unplanned_load: float | None
    unplanned_share: float | None
    current: bool


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _bucket(day: date, origin: date | None) -> date:
    """The week start holding ``day``: the hub cadence when known, Monday otherwise."""
    if origin is None:
        return _monday(day)
    return origin + timedelta(days=7 * ((day - origin).days // 7))


def _family_of(type_name: str) -> frozenset[str]:
    for family in TYPE_FAMILIES:
        if type_name in family:
            return family
    return frozenset({type_name})


def _compatible(planned_type: str | None, activity_type: str) -> bool:
    if planned_type is None or planned_type == activity_type:
        return True
    return activity_type in _family_of(planned_type)


def _is_planned(event: Event) -> bool:
    category = event.category or ""
    return category == "WORKOUT" or category.startswith("RACE_")


def _completion(planned: Event, activity: Activity, *, paired: bool) -> float | None:
    # The hub's compliance scores its own pairing; a fallback match the hub did not
    # score reads the load, duration and distance ratios instead.
    if paired and activity.compliance is not None:
        return activity.compliance / 100.0
    for planned_value, actual_value in (
        (planned.icu_training_load, activity.icu_training_load),
        (planned.moving_time, activity.moving_time),
        (planned.distance, activity.distance),
    ):
        if planned_value and actual_value is not None:
            return actual_value / planned_value
    return None


def _status(completion: float | None) -> Status:
    if completion is None or completion >= COMPLETED_THRESHOLD:
        return "completed"
    if completion >= PARTIAL_THRESHOLD:
        return "partial"
    return "missed"


def match_execution(
    events: Sequence[Event], activities: Sequence[Activity], *, today: date
) -> ExecutionMatch:
    judged_until = today - timedelta(days=1)
    planned = sorted(
        (event for event in events if _is_planned(event)), key=lambda e: e.start_date_local
    )
    usable = list(activities)

    matched: dict[int, Activity] = {}
    paired_flags: dict[int, bool] = {}
    consumed: set[str] = set()

    by_id: dict[str, int] = {}
    for index, event in enumerate(planned):
        if event.id is not None:
            by_id.setdefault(str(event.id), index)
    for activity in sorted(usable, key=lambda a: a.id):
        if activity.paired_event_id is None or activity.id in consumed:
            continue
        paired_index = by_id.get(str(activity.paired_event_id))
        if paired_index is not None and paired_index not in matched:
            matched[paired_index] = activity
            paired_flags[paired_index] = True
            consumed.add(activity.id)

    pairs: list[tuple[int, int, Activity]] = []
    for index, event in enumerate(planned):
        if index in matched or event.start_date_local.date() > today:
            continue
        for activity in usable:
            if activity.paired_event_id is not None or activity.id in consumed:
                continue
            distance = abs((activity.start_date_local.date() - event.start_date_local.date()).days)
            if distance <= 1 and _compatible(event.type, activity.type):
                pairs.append((distance, index, activity))
    pairs.sort(key=lambda item: (item[0], planned[item[1]].start_date_local, item[1], item[2].id))
    for _, index, activity in pairs:
        if index in matched or activity.id in consumed:
            continue
        matched[index] = activity
        consumed.add(activity.id)

    sessions = []
    pending = []
    for index, event in enumerate(planned):
        matched_activity = matched.get(index)
        if matched_activity is None:
            if event.start_date_local.date() >= today:
                pending.append(event)
                continue
            sessions.append(SessionMatch(event, None, "missed", None))
            continue
        paired = paired_flags.get(index, False)
        completion = _completion(event, matched_activity, paired=paired)
        status = _status(completion)
        if status == "missed" and event.start_date_local.date() == today:
            status = "partial"
        sessions.append(SessionMatch(event, matched_activity, status, completion, paired))
    unplanned = tuple(
        sorted(
            (
                activity
                for activity in usable
                if activity.id not in consumed and activity.start_date_local.date() <= judged_until
            ),
            key=lambda a: a.start_date_local,
        )
    )
    return ExecutionMatch(tuple(sessions), unplanned, tuple(pending))


def aggregate_weeks(
    match: ExecutionMatch, *, today: date, origin: date | None = None
) -> tuple[WeekReview, ...]:
    """Fold the judged sessions and unplanned activities into weekly rows.

    Weeks run contiguously on the block cadence when ``origin`` is given (Monday
    otherwise), from the first judged or unplanned day to the later of the current
    week and the last judged day — a session done early can lie in a later week.
    A week with nothing due still shows its row: a prescribed stop is not a
    failure. Loads are None when nothing carries one, adherence is
    ``completed / judged sessions`` and the share reads against all executed
    load. Pending sessions are not part of the fold: nothing is due yet.
    """
    days = [session.planned.start_date_local.date() for session in match.sessions]
    days.extend(activity.start_date_local.date() for activity in match.unplanned)
    if not days:
        return ()
    first = _bucket(min(days), origin)
    last = max(_bucket(today, origin), _bucket(max(days), origin))
    reviews = []
    week_start = first
    while week_start <= last:
        sessions = [
            session
            for session in match.sessions
            if _bucket(session.planned.start_date_local.date(), origin) == week_start
        ]
        unplanned = [
            activity
            for activity in match.unplanned
            if _bucket(activity.start_date_local.date(), origin) == week_start
        ]
        planned_loads = [
            session.planned.icu_training_load
            for session in sessions
            if session.planned.icu_training_load is not None
        ]
        executed_loads = [
            session.activity.icu_training_load
            for session in sessions
            if session.activity is not None and session.activity.icu_training_load is not None
        ]
        unplanned_loads = [
            activity.icu_training_load
            for activity in unplanned
            if activity.icu_training_load is not None
        ]
        completed = sum(1 for session in sessions if session.status == "completed")
        partial = sum(1 for session in sessions if session.status == "partial")
        missed = sum(1 for session in sessions if session.status == "missed")
        executed_total = sum(executed_loads) + sum(unplanned_loads)
        reviews.append(
            WeekReview(
                week_start=week_start,
                planned_load=sum(planned_loads) if planned_loads else None,
                executed_load=sum(executed_loads) if executed_loads else None,
                completed=completed,
                partial=partial,
                missed=missed,
                adherence=completed / len(sessions) if sessions else None,
                unplanned_load=sum(unplanned_loads) if unplanned_loads else None,
                unplanned_share=sum(unplanned_loads) / executed_total
                if executed_total > 0
                else None,
                current=week_start == _bucket(today, origin),
            )
        )
        week_start += timedelta(days=7)
    return tuple(reviews)


@dataclass(frozen=True)
class WeekEfficacy:
    week_start: date
    phase: BlockPhase
    status: BlockWeekStatus
    band_low: float | None
    band_high: float | None
    executed_load: float | None
    fitness: float | None
    ramp: float | None


def compare_trajectory(
    reviews: Sequence[WeekReview],
    block_plan: Sequence[BlockWeek],
    rollup: Sequence[TrainingWeek],
) -> tuple[WeekEfficacy, ...]:
    """Line each judged week up with its adjusted plan and the hub's movement.

    The band and the status are the plan's own (the #64 vocabulary, never
    re-derived); the executed load is the week's total, matched and unplanned;
    fitness and ramp are read from the hub rollup. A reviewed week missing from
    the block (before the block start) is skipped, and a week without a rollup
    row keeps its empty movement. The join reads the exact week starts of the block
    grid and the rows follow the reviews' order.
    """
    weeks = {row.week_start: row for row in block_plan}
    movements = {week.week_start: week for week in rollup}
    rows = []
    for review in reviews:
        planned = weeks.get(review.week_start)
        if planned is None:
            continue
        movement = movements.get(review.week_start)
        loads = [
            value for value in (review.executed_load, review.unplanned_load) if value is not None
        ]
        rows.append(
            WeekEfficacy(
                week_start=review.week_start,
                phase=planned.phase,
                status=planned.status,
                band_low=planned.target_tss_low,
                band_high=planned.target_tss_high,
                executed_load=sum(loads) if loads else None,
                fitness=movement.fitness if movement is not None else None,
                ramp=movement.ramp_rate if movement is not None else None,
            )
        )
    return tuple(rows)


def build_review(
    match: ExecutionMatch,
    weeks: Sequence[WeekReview],
    efficacy: Sequence[WeekEfficacy],
) -> ExecutionReview:
    """The compact digest: one row per judged block week, one line per session.

    Every judged session keeps its exact planned/actual numbers whatever its
    status, and unplanned activities are listed with their own numbers; the
    week rows only cover weeks present in the adjusted block plan.
    """
    by_start = {row.week_start: row for row in efficacy}
    week_rows = [
        ReviewWeek(
            week_start=week.week_start,
            phase=efficacy_by_start.phase,
            status=efficacy_by_start.status,
            band_low=efficacy_by_start.band_low,
            band_high=efficacy_by_start.band_high,
            planned_load=week.planned_load,
            executed_load=efficacy_by_start.executed_load,
            completed=week.completed,
            partial=week.partial,
            missed=week.missed,
            adherence=week.adherence,
            unplanned_load=week.unplanned_load,
            unplanned_share=week.unplanned_share,
            fitness=efficacy_by_start.fitness,
            ramp=efficacy_by_start.ramp,
            current=week.current,
        )
        for week in weeks
        if (efficacy_by_start := by_start.get(week.week_start)) is not None
    ]
    sessions = [
        ReviewSession(
            date=session.planned.start_date_local.date(),
            status=session.status,
            planned=session.planned.name,
            planned_load=session.planned.icu_training_load,
            planned_time_s=session.planned.moving_time,
            planned_distance_m=session.planned.distance,
            actual=session.activity.name if session.activity is not None else None,
            actual_load=session.activity.icu_training_load
            if session.activity is not None
            else None,
            actual_time_s=session.activity.moving_time if session.activity is not None else None,
            actual_distance_m=session.activity.distance if session.activity is not None else None,
            compliance=session.activity.compliance
            if session.activity is not None and session.paired
            else None,
        )
        for session in match.sessions
    ]
    sessions.extend(
        ReviewSession(
            date=activity.start_date_local.date(),
            status="unplanned",
            actual=activity.name,
            actual_load=activity.icu_training_load,
            actual_time_s=activity.moving_time,
            actual_distance_m=activity.distance,
            compliance=None,
        )
        for activity in match.unplanned
    )
    sessions.sort(key=lambda session: session.date)
    return ExecutionReview(weeks=week_rows, sessions=sessions)

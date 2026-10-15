"""Time-invariant periodization skeleton for the race block.

A pure function of the anchor race, the weekly grid and the executed history:
the same inputs derive the same weeks, recomputed every turn and never persisted.
Only ``today`` positions the current week; it never moves the block. See
docs/periodization.md for the doctrine and the design decisions.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from open_endurance_coach.schemas.context import (
    BlockPhase,
    BlockWeek,
    BlockWeekStatus,
    GoalRace,
    TrainingWeek,
)

PREP_WEEKS = 2
BASE_WEEKS = 12
BASE_FLOOR_WEEKS = 8
BUILD_WEEKS = 8
BUILD_FLOOR_WEEKS = 4
PEAK_WEEKS = 2
MESOCYCLE_WEEKS = 4
CLOSE_A_RACES_DAYS = 21
NOMINAL_TRAINING_WEEKS = PREP_WEEKS + BASE_WEEKS + BUILD_WEEKS + PEAK_WEEKS

# 6 x 7: weekly TSS for one CTL point per week under the 42-day CTL model.
WEEKLY_TSS_PER_CTL = 42.0
SUSTAINED_RAMP = (5.0, 8.0)
BUILD_RAMP = (0.0, 4.0)
PREP_RAMP = (0.0, 4.0)
CRASH_RAMP = (8.0, 10.0)
RECOVERY_RAMP = (-5.0, -2.0)
PEAK_RAMP = (-4.0, 0.0)
PEAK_TAPER_RAMP = (-6.0, -3.0)
RACE_CTL_LOSS = 0.1

_CATEGORY_RANK = {"RACE_A": 0, "RACE_B": 1, "RACE_C": 2}

_PREP_ROLE = "Prep: reintroduce aerobic work; volume over intensity."
_BASE_ROLE = "Base: aerobic volume, Z2; one long session."
_HYBRID_BASE_ROLE = (
    "Hybrid Base: Z3 / sweet spot from the first week (muscular endurance);"
    " linear volume accumulation is not possible."
)
_RECOVERY_ROLE = (
    "Recovery tail: 2-3 short Z1 days + 2-3 short intensity openers; load drops naturally."
)
_BUILD_ROLE = "Build: small load steps (0-4 CTL); one long session and one threshold session."
_PEAK_ROLE = (
    "Peak: race-like sessions every 72 h; Z1-only recovery; aim at a positive TSB on race day."
)
_RACE_ROLE = "Race week: short openers; keep CTL within 10% of race day."
_REST_ROLE = "Rest: stopped training; no planned load, resume without making up the missed load."
_OMITTED_ROLE = "Omitted: the block loses this week to resynchronize after the interruption."
_REPEAT_ROLE = "Repeat: redo the last hard week; resume only when it goes well."
_STEP_BACK_BUILD_ROLE = (
    "Step back to Base 3: aerobic rebuild; resume in Z1-2 until heart rate and perceived"
    " effort are back to normal."
)
_STEP_BACK_BASE_ROLE = (
    "Step back to Base 1: aerobic rebuild; resume in Z1-2 until heart rate and perceived"
    " effort are back to normal."
)
_STEP_BACK_PREP_ROLE = (
    "Step back to Prep (back up one block): reintroduce aerobic work; resume in Z1-2 until"
    " heart rate and perceived effort are back to normal."
)
_STEP_BACK_ILLNESS_ROLE = (
    "Step back to Prep (illness): resume in Z1-2 until heart rate and perceived effort are"
    " back to normal."
)


@dataclass(frozen=True)
class PhaseTemplate:
    """The K.I.S.S. template compressed to the available training weeks."""

    prep: int
    base: int
    build: int
    peak: int
    hybrid: bool = False

    @property
    def training_weeks(self) -> int:
        return self.prep + self.base + self.build + self.peak


def template_for(available_weeks: int) -> PhaseTemplate:
    """Compress Prep 2 / Base 12 / Build 8 / Peak 2 to ``available_weeks``.

    Cascade: drop Prep below 24 weeks, shrink Base to its floor of 8, then Build
    to its floor of 4 (protecting the Base floor); below the floors the Base turns
    hybrid (early Z3 / sweet spot) and finally shortens, keeping a single Peak week
    as long as one training week remains.
    """
    prep, base, build, peak = PREP_WEEKS, BASE_WEEKS, BUILD_WEEKS, PEAK_WEEKS
    total = prep + base + build + peak
    if total > available_weeks:
        prep = 0
        total -= PREP_WEEKS
    if total > available_weeks:
        cut = min(total - available_weeks, base - BASE_FLOOR_WEEKS)
        base -= cut
        total -= cut
    if total > available_weeks:
        cut = min(total - available_weeks, build - BUILD_FLOOR_WEEKS)
        build -= cut
        total -= cut
    if total > available_weeks:
        cut = min(total - available_weeks, peak - 1)
        peak -= cut
        total -= cut
    hybrid = False
    if total > available_weeks:
        hybrid = True
        cut = min(total - available_weeks, base)
        base -= cut
        total -= cut
    if total > available_weeks:
        cut = min(total - available_weeks, build)
        build -= cut
        total -= cut
    if total > available_weeks:
        peak = 0
    return PhaseTemplate(prep=prep, base=base, build=build, peak=peak, hybrid=hybrid)


def select_anchor(races: Sequence[GoalRace]) -> GoalRace | None:
    """The race imposing the maximal constraint: nearest A, then nearest race."""
    if not races:
        return None
    return min(races, key=lambda race: (_CATEGORY_RANK.get(race.category, 9), race.date))


def _close_a_races(anchor: GoalRace, races: Sequence[GoalRace]) -> list[GoalRace]:
    if anchor.category != "RACE_A":
        return []
    return sorted(
        (
            race
            for race in races
            if race.category == "RACE_A"
            and race is not anchor
            and 0 < (race.date - anchor.date).days < CLOSE_A_RACES_DAYS
        ),
        key=lambda race: race.date,
    )


def _weekday_grid(weeks: Sequence[TrainingWeek]) -> date | None:
    """The hub's week-start cadence (Monday in practice), None when history is empty."""
    return min(week.week_start for week in weeks) if weeks else None


def _week_start(day: date, cadence: date | None) -> date:
    if cadence is None:
        return day - timedelta(days=day.weekday())
    return day - timedelta(days=(day - cadence).days % 7)


def _executed(week: TrainingWeek) -> bool:
    return week.sessions > 0 or (week.load or 0.0) > 0.0


def _load_of(week: TrainingWeek) -> float:
    return week.load or 0.0


def _latest_ctl(weeks: Sequence[TrainingWeek]) -> float | None:
    for week in reversed(weeks):
        if week.fitness is not None and math.isfinite(week.fitness):
            return week.fitness
    return None


def _rounded(value: float) -> float:
    return round(value / 5.0) * 5.0


def _band(ctl: float | None, ramp: tuple[float, float]) -> tuple[float, float] | None:
    if ctl is None:
        return None
    maintenance = 7.0 * ctl
    low = max(0.0, _rounded(maintenance + WEEKLY_TSS_PER_CTL * ramp[0]))
    high = max(low, _rounded(maintenance + WEEKLY_TSS_PER_CTL * ramp[1]))
    return (low, high)


def _race_week_band(ctl: float | None) -> tuple[float, float] | None:
    if ctl is None:
        return None
    maintenance = 7.0 * ctl
    floor = maintenance - WEEKLY_TSS_PER_CTL * RACE_CTL_LOSS * ctl
    return (_rounded(floor), _rounded(maintenance))


def _is_recovery_tail(number: int, length: int) -> bool:
    """Every fourth week, plus the closing week when its mesocycle has two or more.

    A one-week remainder joins the previous mesocycle: recovery weeks are never
    back to back.
    """
    if number % MESOCYCLE_WEEKS == 0:
        return True
    return number == length and length % MESOCYCLE_WEEKS in (2, 3)


def _crash_week(base_length: int) -> int | None:
    mesocycles = base_length // MESOCYCLE_WEEKS
    if mesocycles == 0:
        return None
    return (mesocycles - 1) * MESOCYCLE_WEEKS + 3


def _base_role(number: int, length: int, hybrid: bool) -> str:
    if _is_recovery_tail(number, length):
        return _RECOVERY_ROLE
    return _HYBRID_BASE_ROLE if hybrid else _BASE_ROLE


def _phase_band(
    phase: str,
    number: int,
    length: int,
    ctl: float | None,
    *,
    elite: bool,
) -> tuple[float, float] | None:
    if phase == "Race":
        return _race_week_band(ctl)
    if phase == "Prep":
        return _band(ctl, PREP_RAMP)
    if phase == "Peak":
        return _band(ctl, PEAK_TAPER_RAMP if number == length else PEAK_RAMP)
    if phase == "Build":
        return _band(ctl, RECOVERY_RAMP if _is_recovery_tail(number, length) else BUILD_RAMP)
    if _is_recovery_tail(number, length):
        return _band(ctl, RECOVERY_RAMP)
    crash = elite and number == _crash_week(length)
    return _band(ctl, CRASH_RAMP if crash else SUSTAINED_RAMP)


def _lay_out(
    template: PhaseTemplate,
    block_start: date,
    *,
    ctl: float | None,
    elite: bool,
) -> list[BlockWeek]:
    phases: list[tuple[BlockPhase, int]] = [
        ("Prep", template.prep),
        ("Base", template.base),
        ("Build", template.build),
        ("Peak", template.peak),
        ("Race", 1),
    ]
    rows: list[BlockWeek] = []
    cursor = block_start
    for phase, length in phases:
        for number in range(1, length + 1):
            if phase == "Prep":
                role = _PREP_ROLE
            elif phase == "Base":
                role = _base_role(number, length, template.hybrid)
            elif phase == "Build":
                role = _RECOVERY_ROLE if _is_recovery_tail(number, length) else _BUILD_ROLE
            elif phase == "Peak":
                role = _PEAK_ROLE
            else:
                role = _RACE_ROLE
            band = _phase_band(phase, number, length, ctl, elite=elite)
            rows.append(
                BlockWeek(
                    week_start=cursor,
                    phase=phase,
                    week_in_phase=number,
                    role=role,
                    target_tss_low=band[0] if band else None,
                    target_tss_high=band[1] if band else None,
                    status="upcoming",
                )
            )
            cursor += timedelta(days=7)
    return rows


def _race_suffix(race: GoalRace, *, target: bool, prolonged: bool) -> str:
    if target:
        return f" Target race: {race.name} ({race.category})."
    if prolonged:
        return f" A race {race.name}: prolonged peak covering both races."
    if race.category == "RACE_B":
        return f" B race {race.name}: mini-taper 2-4 days."
    if race.category == "RACE_C":
        return f" C race {race.name}: trained through."
    return f" A race {race.name}."


def _mark_races(
    rows: Sequence[BlockWeek],
    races: Sequence[GoalRace],
    *,
    cadence: date | None,
    target: GoalRace,
    prolonged: Sequence[GoalRace],
) -> None:
    """Name every race of the best priority in its week; the target comes first."""
    by_start = {row.week_start: row for row in rows}
    base_roles = {row.week_start: row.role for row in rows}
    groups: dict[date, list[GoalRace]] = {}
    for race in races:
        start = _week_start(race.date, cadence)
        if start in by_start:
            groups.setdefault(start, []).append(race)
    for start, group in groups.items():

        def rank(race: GoalRace) -> int:
            if race is target or any(race is other for other in prolonged):
                return 0
            return _CATEGORY_RANK.get(race.category, 9)

        best = min(rank(race) for race in group)
        selected = [race for race in group if rank(race) == best]
        selected.sort(key=lambda race: (0 if race is target else 1, race.date, race.name))
        row = by_start[start]
        row.role = base_roles[start] + "".join(
            _race_suffix(
                race,
                target=race is target,
                prolonged=any(race is other for other in prolonged),
            )
            for race in selected
        )


def _executed_status(
    band_low: float | None, band_high: float | None, load: float
) -> BlockWeekStatus:
    if band_low is None or band_high is None:
        return "done" if load > 0.0 else "under"
    if load < band_low:
        return "under"
    if load > band_high:
        return "over"
    return "done"


def _apply_statuses(rows: Sequence[BlockWeek], executed: dict[date, float], *, today: date) -> None:
    for row in rows:
        week_end = row.week_start + timedelta(days=6)
        if week_end < today:
            row.status = _executed_status(
                row.target_tss_low, row.target_tss_high, executed.get(row.week_start, 0.0)
            )
        elif row.week_start <= today:
            row.status = "current"
        else:
            row.status = "upcoming"


def _phase_length(rows: Sequence[BlockWeek], phase: str) -> int:
    return max((row.week_in_phase for row in rows if row.phase == phase), default=0)


def _is_late_cut(rows: Sequence[BlockWeek], current: BlockWeek) -> bool:
    """Close enough to the race that restructuring the block would hurt it."""
    if current.phase in ("Peak", "Race"):
        return True
    build = _phase_length(rows, "Build")
    return current.phase == "Build" and build >= BUILD_WEEKS and current.week_in_phase == build


def _step_back_role(phase: str, *, illness: bool, deep: bool) -> str:
    if phase == "Build":
        return _STEP_BACK_BUILD_ROLE
    if deep:
        return _STEP_BACK_PREP_ROLE
    if illness:
        return _STEP_BACK_ILLNESS_ROLE
    return _STEP_BACK_BASE_ROLE


def _entry_band(phase: str, ctl: float | None) -> tuple[float, float] | None:
    return _band(ctl, PREP_RAMP if phase == "Prep" else SUSTAINED_RAMP)


def _mark_rest(row: BlockWeek) -> None:
    row.status = "rest"
    row.role = _REST_ROLE
    row.target_tss_low = 0.0
    row.target_tss_high = 0.0


def _mark_omitted(row: BlockWeek) -> None:
    row.status = "omitted"
    row.role = _OMITTED_ROLE
    row.target_tss_low = 0.0
    row.target_tss_high = 0.0


def _omittable(row: BlockWeek, current_week: date) -> bool:
    return (
        row.week_start > current_week
        and row.status in ("current", "upcoming")
        and row.phase != "Race"
    )


def _omissions_4_7(
    rows: Sequence[BlockWeek], current: BlockWeek, current_week: date
) -> list[BlockWeek]:
    """Reduce the current block by one week; in the Peak, Peak 2 becomes 1.

    In the Peak the stopped week itself absorbs the reduction (it is statused
    rest), so nothing is omitted and the race-eve taper week is never dropped.
    """
    if current.phase == "Peak":
        return []
    peak = [row for row in rows if row.phase == "Peak"]
    candidates: list[BlockWeek] = []
    phase_rows = [
        row for row in rows if row.phase == current.phase and _omittable(row, current_week)
    ]
    if phase_rows:
        candidates.append(phase_rows[-1])
    if len(peak) == 2:
        candidates.append(peak[0])
    return candidates


def _omissions_8_14(rows: Sequence[BlockWeek]) -> list[BlockWeek]:
    candidates: list[BlockWeek] = []
    build_second = [row for row in rows if row.phase == "Build" and row.week_in_phase == 2]
    peak_first = [row for row in rows if row.phase == "Peak" and row.week_in_phase == 1]
    candidates.extend(build_second)
    candidates.extend(peak_first)
    return candidates


def _omissions_deep(rows: Sequence[BlockWeek]) -> list[BlockWeek]:
    candidates: list[BlockWeek] = []
    peak_first = [row for row in rows if row.phase == "Peak" and row.week_in_phase == 1]
    build_second = [row for row in rows if row.phase == "Build" and row.week_in_phase == 2]
    build_first = [row for row in rows if row.phase == "Build" and row.week_in_phase == 1]
    candidates.extend(peak_first)
    candidates.extend(build_second)
    candidates.extend(build_first)
    return candidates


def _apply_omissions(candidates: Iterable[BlockWeek], *, current_week: date, count: int) -> None:
    omitted = 0
    for row in candidates:
        if omitted >= count:
            return
        if not _omittable(row, current_week):
            continue
        _mark_omitted(row)
        omitted += 1


def _apply_contingency(
    rows: Sequence[BlockWeek],
    trained: dict[date, bool],
    *,
    today: date,
    illness: bool,
    ctl: float | None,
) -> None:
    if not rows:
        return
    cadence = rows[0].week_start
    current_week = _week_start(today, cadence)
    if not any(trained.values()):
        return
    current_executed = trained.get(current_week, False)
    elapsed = (today - current_week).days
    empty_full = 0
    floor = min(trained)
    cursor = current_week - timedelta(days=7)
    while cursor >= floor and not trained.get(cursor, False):
        empty_full += 1
        cursor -= timedelta(days=7)
    gap_days = 7 * empty_full + (0 if current_executed else elapsed)
    if gap_days <= 3:
        return
    by_start = {row.week_start: row for row in rows}
    stopped = [current_week - timedelta(days=7 * offset) for offset in range(1, empty_full + 1)]
    if not current_executed and elapsed >= 3:
        stopped.insert(0, current_week)
    for start in stopped:
        row = by_start.get(start)
        if row is not None and row.phase != "Race":
            _mark_rest(row)
    current = by_start.get(current_week)
    if current is None:
        return
    if gap_days <= 7 and not illness:
        _apply_omissions(
            _omissions_4_7(rows, current, current_week), current_week=current_week, count=1
        )
        return
    if _is_late_cut(rows, current):
        return
    deep = gap_days > 14
    current.status = "step_back"
    current.role = _step_back_role(current.phase, illness=illness, deep=deep)
    prep_entry = current.phase != "Build" and (illness or deep)
    band = _entry_band("Prep" if prep_entry else current.phase, ctl)
    current.target_tss_low = band[0] if band else None
    current.target_tss_high = band[1] if band else None
    if not deep:
        following = by_start.get(current_week + timedelta(days=7))
        if following is not None and following.phase != "Race":
            following.status = "repeat"
            following.role = _REPEAT_ROLE
        _apply_omissions(_omissions_8_14(rows), current_week=current_week, count=1)
    else:
        _apply_omissions(_omissions_deep(rows), current_week=current_week, count=2)


def derive_block_plan(
    races: Sequence[GoalRace],
    weeks: Sequence[TrainingWeek],
    *,
    today: date,
    illness: bool = False,
    elite: bool = False,
) -> list[BlockWeek]:
    """Derive the block for the anchored race; empty when no race anchors it.

    The block is race-anchored: it starts at the later of the first executed week
    and 24 training weeks before the race, so it never moves as ``today`` advances
    and past weeks are never re-labelled. Only ``today`` positions the current week
    and the execution and interruption statuses.

    ``elite`` and ``illness`` are declared by the caller from the athlete's own words;
    no metric threshold opens them. The hub path passes neither until the declared
    facts are wired, so the crash week and the illness branch fire only when a caller
    declares them.
    """
    anchor = select_anchor(races)
    if anchor is None:
        return []
    close = _close_a_races(anchor, races)
    block_race = close[-1] if close else anchor
    prolonged = [anchor, *close[:-1]] if close else []
    cadence = _weekday_grid(weeks)
    race_week = _week_start(block_race.date, cadence)
    first_executed = min((week.week_start for week in weeks if _executed(week)), default=None)
    nominal_start = race_week - timedelta(days=7 * NOMINAL_TRAINING_WEEKS)
    block_start = max(first_executed, nominal_start) if first_executed else nominal_start
    available = max(0, (race_week - block_start).days // 7)
    template = template_for(available)
    block_start = race_week - timedelta(days=7 * template.training_weeks)
    ctl = _latest_ctl(weeks)
    rows = _lay_out(template, block_start, ctl=ctl, elite=elite)
    _mark_races(
        rows,
        races,
        cadence=cadence,
        target=block_race,
        prolonged=prolonged,
    )
    executed = {week.week_start: _load_of(week) for week in weeks}
    _apply_statuses(rows, executed, today=today)
    trained = {week.week_start: _executed(week) for week in weeks}
    _apply_contingency(rows, trained, today=today, illness=illness, ctl=ctl)
    return rows

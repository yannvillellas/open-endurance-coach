from datetime import date, timedelta

import pytest

from open_endurance_coach.extractors.budget import build_within_budget
from open_endurance_coach.extractors.context import ContextPlan, gather_context
from open_endurance_coach.extractors.periodization import (
    PhaseTemplate,
    derive_block_plan,
    select_anchor,
    template_for,
)
from open_endurance_coach.schemas.context import BlockWeek, CoachContext, GoalRace, TrainingWeek
from open_endurance_coach.schemas.intervals import Activity, Wellness

from .fakes import make_activity, make_intervals_client, make_wellness

TODAY = date(2024, 2, 5)  # a Monday
RACE_DAY = date(2024, 6, 17)  # a Monday, 19 weeks out
PREP_START = date(2024, 1, 1)  # race day minus 24 weeks


def make_race(
    day: date,
    *,
    category: str = "RACE_A",
    name: str = "Summer A",
    event_id: int = 1,
) -> GoalRace:
    days = (day - TODAY).days
    return GoalRace(
        event_id=event_id,
        name=name,
        date=day,
        category=category,
        days_to_race=days,
        weeks_to_race=(days + 6) // 7,
    )


def make_week(
    day: date,
    *,
    load: float | None = 300.0,
    sessions: int = 4,
    fitness: float | None = 50.0,
) -> TrainingWeek:
    return TrainingWeek(week_start=day, sessions=sessions, time_s=7200, load=load, fitness=fitness)


def history_until(
    last: date,
    *,
    first: date = PREP_START,
    load: float | None = 300.0,
    fitness: float | None = 50.0,
) -> list[TrainingWeek]:
    weeks: list[TrainingWeek] = []
    cursor = first
    while cursor <= last:
        weeks.append(make_week(cursor, load=load, fitness=fitness))
        cursor += timedelta(days=7)
    return weeks


def by_start(rows: list[BlockWeek]) -> dict[date, BlockWeek]:
    return {row.week_start: row for row in rows}


def test_empty_races_derive_no_block() -> None:
    assert derive_block_plan([], history_until(TODAY), today=TODAY) == []


def test_the_block_ends_on_the_anchor_race_week() -> None:
    rows = derive_block_plan([make_race(RACE_DAY)], [], today=TODAY)

    assert rows[0].week_start == PREP_START
    assert rows[0].phase == "Prep"
    assert rows[0].week_in_phase == 1
    assert rows[-1].week_start == RACE_DAY
    assert rows[-1].phase == "Race"
    assert {row.week_start.weekday() for row in rows} == {0}
    assert [row.week_start for row in rows] == sorted(row.week_start for row in rows)


def test_anchor_prefers_the_nearest_a_race_over_a_closer_b_race() -> None:
    races = [
        make_race(date(2024, 2, 12), category="RACE_B", name="Club Race", event_id=2),
        make_race(RACE_DAY, name="Summer A", event_id=1),
    ]
    rows = by_start(derive_block_plan(races, history_until(TODAY), today=TODAY))

    assert rows[RACE_DAY].role.startswith("Race week")
    assert "Target race: Summer A" in rows[RACE_DAY].role
    assert "B race Club Race: mini-taper 2-4 days." in rows[date(2024, 2, 12)].role


def test_anchor_falls_back_to_the_nearest_race_by_priority() -> None:
    races = [
        make_race(date(2024, 3, 11), category="RACE_C", name="Local CX", event_id=3),
        make_race(date(2024, 3, 4), category="RACE_B", name="Club B", event_id=2),
    ]
    rows = derive_block_plan(races, history_until(TODAY), today=TODAY)

    assert rows[-1].week_start == date(2024, 3, 4)
    assert "Target race: Club B" in rows[-1].role
    assert not any("Local CX" in row.role for row in rows)


def test_a_c_race_anchors_when_no_a_or_b_race_supports_the_block() -> None:
    races = [make_race(date(2024, 3, 11), category="RACE_C", name="Local CX", event_id=3)]
    rows = derive_block_plan(races, history_until(TODAY), today=TODAY)

    assert rows[-1].week_start == date(2024, 3, 11)
    assert "Target race: Local CX" in rows[-1].role


def test_select_anchor_ranks_the_category_before_the_date() -> None:
    race_a = make_race(RACE_DAY, name="Far A", event_id=1)
    race_b = make_race(date(2024, 2, 12), category="RACE_B", name="Near B", event_id=2)

    assert select_anchor([race_b, race_a]) is race_a
    assert select_anchor([]) is None


def test_an_a_race_within_three_weeks_prolongs_the_peak() -> None:
    first = make_race(date(2024, 6, 10), name="A One", event_id=1)
    second = make_race(date(2024, 6, 24), name="A Two", event_id=2)
    rows = by_start(derive_block_plan([first, second], history_until(TODAY), today=TODAY))

    assert max(rows) == date(2024, 6, 24)
    assert rows[date(2024, 6, 24)].phase == "Race"
    assert "Target race: A Two" in rows[date(2024, 6, 24)].role
    assert "A One" in rows[date(2024, 6, 10)].role
    assert "prolonged peak" in rows[date(2024, 6, 10)].role


def test_a_same_week_a_pair_names_the_target_first() -> None:
    first = make_race(date(2024, 6, 17), name="A One", event_id=1)
    second = make_race(date(2024, 6, 20), name="A Two", event_id=2)
    rows = by_start(derive_block_plan([first, second], history_until(TODAY), today=TODAY))

    role = rows[date(2024, 6, 17)].role
    assert "Target race: A Two" in role
    assert "A One" in role and "prolonged peak" in role
    assert role.index("A Two") < role.index("A One")


def test_a_second_a_race_weeks_later_is_not_detailed() -> None:
    first = make_race(date(2024, 6, 10), name="A One", event_id=1)
    second = make_race(date(2024, 7, 15), name="A Two", event_id=2)
    rows = derive_block_plan([first, second], history_until(TODAY), today=TODAY)

    assert rows[-1].week_start == date(2024, 6, 10)
    assert all("A Two" not in row.role for row in rows)


def test_continuous_history_keeps_the_race_anchored_block_start() -> None:
    rows = derive_block_plan(
        [make_race(RACE_DAY)],
        history_until(TODAY, first=PREP_START - timedelta(days=28)),
        today=TODAY,
    )

    assert rows[0].week_start == PREP_START
    assert rows[0].phase == "Prep"


def test_a_recent_start_compresses_the_block_to_the_available_weeks() -> None:
    first = date(2024, 1, 22)  # 21 training weeks before the race
    rows = derive_block_plan([make_race(RACE_DAY)], history_until(TODAY, first=first), today=TODAY)

    assert rows[0].week_start == first
    assert rows[0].phase == "Base"
    assert rows[0].week_in_phase == 1
    assert not any(row.phase == "Prep" for row in rows)


def test_a_short_horizon_turns_the_base_hybrid() -> None:
    today = date(2024, 4, 29)
    first = date(2024, 3, 25)  # 12 training weeks before the race
    rows = derive_block_plan([make_race(RACE_DAY)], history_until(today, first=first), today=today)

    assert rows[0].week_start == first
    assert rows[0].phase == "Base"
    assert "Hybrid Base" in rows[0].role
    assert all(row.phase != "Prep" for row in rows)


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (24, (2, 12, 8, 2, False)),
        (23, (0, 12, 8, 2, False)),
        (22, (0, 12, 8, 2, False)),
        (21, (0, 11, 8, 2, False)),
        (19, (0, 9, 8, 2, False)),
        (18, (0, 8, 8, 2, False)),
        (17, (0, 8, 7, 2, False)),
        (14, (0, 8, 4, 2, False)),
        (13, (0, 8, 4, 1, False)),
        (12, (0, 7, 4, 1, True)),
        (10, (0, 5, 4, 1, True)),
        (4, (0, 0, 3, 1, True)),
        (1, (0, 0, 0, 1, True)),
        (0, (0, 0, 0, 0, True)),
    ],
)
def test_the_cascade_compresses_24_12_8_4(
    available: int, expected: tuple[int, int, int, int, bool]
) -> None:
    template = template_for(available)

    assert template == PhaseTemplate(*expected)


def test_the_grid_follows_the_rollup_week_starts() -> None:
    sunday = date(2024, 2, 4)
    weeks = [make_week(sunday + timedelta(days=7 * index)) for index in range(2)]
    rows = derive_block_plan([make_race(RACE_DAY)], weeks, today=TODAY)

    assert {row.week_start.weekday() for row in rows} == {6}
    assert rows[-1].week_start == date(2024, 6, 16)  # Sunday before the race Monday


def test_the_template_reproduces_a_nominal_block() -> None:
    rows = derive_block_plan([make_race(RACE_DAY)], history_until(TODAY), today=TODAY)

    assert [row.phase for row in rows[:4]] == ["Prep", "Prep", "Base", "Base"]
    assert [row.phase for row in rows[-4:]] == ["Build", "Peak", "Peak", "Race"]
    assert len(rows) == 25


def test_recomputing_with_the_same_inputs_is_deterministic() -> None:
    race = make_race(RACE_DAY)
    history = history_until(TODAY)

    assert derive_block_plan([race], history, today=TODAY) == derive_block_plan(
        [race], history, today=TODAY
    )


def test_a_later_recomputation_keeps_past_weeks() -> None:
    race = make_race(RACE_DAY)
    plan_before = derive_block_plan([race], history_until(TODAY), today=TODAY)
    later = date(2024, 2, 12)
    plan_after = derive_block_plan([race], history_until(later), today=later)

    before = by_start(plan_before)
    after = by_start(plan_after)
    assert set(before) == set(after)
    for start, row in before.items():
        other = after[start]
        assert (
            other.phase,
            other.week_in_phase,
            other.role,
            other.target_tss_low,
            other.target_tss_high,
        ) == (row.phase, row.week_in_phase, row.role, row.target_tss_low, row.target_tss_high)
        if start + timedelta(days=6) < TODAY:
            assert other.status == row.status


def test_past_weeks_are_judged_against_the_band() -> None:
    history = [
        make_week(date(2024, 1, 1), load=350.0),
        make_week(date(2024, 1, 8), load=300.0),
        make_week(date(2024, 1, 15), load=600.0),
        make_week(date(2024, 1, 22), load=500.0),
        make_week(date(2024, 1, 29), load=700.0),
        make_week(TODAY, load=120.0),
    ]
    rows = by_start(derive_block_plan([make_race(RACE_DAY)], history, today=TODAY))

    assert rows[date(2024, 1, 1)].status == "done"
    assert rows[date(2024, 1, 8)].status == "under"
    assert rows[date(2024, 1, 15)].status == "done"
    assert rows[date(2024, 1, 22)].status == "under"
    assert rows[date(2024, 1, 29)].status == "over"
    assert rows[TODAY].status == "current"
    assert rows[date(2024, 2, 12)].status == "upcoming"
    assert rows[RACE_DAY].status == "upcoming"


def test_bands_read_the_hub_ctl() -> None:
    rows = by_start(derive_block_plan([make_race(RACE_DAY)], history_until(TODAY), today=TODAY))

    def band(day: date) -> tuple[float | None, float | None]:
        return (rows[day].target_tss_low, rows[day].target_tss_high)

    assert band(date(2024, 1, 1)) == (350.0, 520.0)  # Prep, +0 to +4 CTL/week
    assert band(date(2024, 1, 15)) == (560.0, 685.0)  # Base, +5 to +8 CTL/week
    assert band(date(2024, 2, 5)) == (140.0, 265.0)  # Base recovery tail
    assert band(date(2024, 4, 8)) == (350.0, 520.0)  # Build, +0 to +4
    assert band(date(2024, 4, 29)) == (140.0, 265.0)  # Build recovery tail
    assert band(date(2024, 6, 3)) == (180.0, 350.0)  # Peak
    assert band(date(2024, 6, 10)) == (100.0, 225.0)  # Peak taper week
    assert band(RACE_DAY) == (140.0, 350.0)  # race week within 10% CTL loss


def test_a_non_finite_ctl_from_the_hub_is_ignored() -> None:
    nan_week = make_week(TODAY, load=120.0, fitness=float("nan"))
    inf_week = make_week(date(2024, 1, 29), load=120.0, fitness=float("inf"))
    rows = by_start(
        derive_block_plan(
            [make_race(RACE_DAY)],
            [*history_until(date(2024, 1, 22)), inf_week, nan_week],
            today=TODAY,
        )
    )
    assert (rows[date(2024, 1, 15)].target_tss_low, rows[date(2024, 1, 15)].target_tss_high) == (
        560.0,
        685.0,
    )

    without_ctl = by_start(
        derive_block_plan(
            [make_race(RACE_DAY)],
            [make_week(date(2024, 1, 22), fitness=float("inf")), nan_week],
            today=TODAY,
        )
    )
    assert without_ctl[RACE_DAY].target_tss_low is None
    assert without_ctl[RACE_DAY].target_tss_high is None


def test_the_crash_week_needs_a_declared_elite_athlete() -> None:
    regular = by_start(
        derive_block_plan([make_race(RACE_DAY)], history_until(TODAY, fitness=50.0), today=TODAY)
    )
    high_ctl = by_start(
        derive_block_plan([make_race(RACE_DAY)], history_until(TODAY, fitness=90.0), today=TODAY)
    )
    declared_elite = by_start(
        derive_block_plan(
            [make_race(RACE_DAY)],
            history_until(TODAY, fitness=50.0),
            today=TODAY,
            elite=True,
        )
    )

    crash = date(2024, 3, 25)  # Base 11, the third week of the last full mesocycle
    assert (regular[crash].target_tss_low, regular[crash].target_tss_high) == (560.0, 685.0)
    # A high hub CTL alone never opens the gate: the sustained 5-8 band applies.
    assert (high_ctl[crash].target_tss_low, high_ctl[crash].target_tss_high) == (840.0, 965.0)
    assert (declared_elite[crash].target_tss_low, declared_elite[crash].target_tss_high) == (
        685.0,
        770.0,
    )
    assert declared_elite[date(2024, 3, 18)].target_tss_high == 685.0  # one crash week only


def test_mesocycles_end_in_recovery_tails() -> None:
    rows = by_start(derive_block_plan([make_race(RACE_DAY)], history_until(TODAY), today=TODAY))

    assert "Recovery tail" in rows[date(2024, 2, 5)].role  # Base 4
    assert "Recovery tail" in rows[date(2024, 3, 4)].role  # Base 8
    assert "Recovery tail" in rows[date(2024, 4, 1)].role  # Base 12
    assert "Recovery tail" in rows[date(2024, 4, 29)].role  # Build 4
    assert "Recovery tail" in rows[date(2024, 5, 27)].role  # Build 8
    assert "Recovery tail" not in rows[date(2024, 1, 29)].role  # Base 3


def test_a_short_interruption_marks_rest_and_omits_one_week() -> None:
    today = date(2024, 2, 11)  # Sunday, the current week has no session yet
    rows = by_start(
        derive_block_plan([make_race(RACE_DAY)], history_until(date(2024, 1, 29)), today=today)
    )

    assert rows[TODAY].status == "rest"
    assert (rows[TODAY].target_tss_low, rows[TODAY].target_tss_high) == (0.0, 0.0)
    assert rows[date(2024, 4, 1)].status == "omitted"  # Base 12, the current phase's last week
    assert rows[date(2024, 2, 12)].status == "upcoming"


def test_a_rest_week_in_the_peak_never_omits_the_taper_week() -> None:
    today = date(2024, 6, 9)  # Sunday of the Peak 1 week, no session yet
    rows = by_start(
        derive_block_plan([make_race(RACE_DAY)], history_until(date(2024, 5, 27)), today=today)
    )

    assert rows[date(2024, 6, 3)].status == "rest"
    assert rows[date(2024, 6, 10)].status == "upcoming"


def test_a_full_week_off_is_a_rest_week_on_resume() -> None:
    today = date(2024, 2, 12)
    history = [
        *history_until(date(2024, 1, 29)),
        make_week(date(2024, 2, 5), load=0.0, sessions=0),
        make_week(today, load=100.0),
    ]
    rows = by_start(derive_block_plan([make_race(RACE_DAY)], history, today=today))

    assert rows[date(2024, 2, 5)].status == "rest"
    assert rows[today].status == "current"
    assert rows[date(2024, 4, 1)].status == "omitted"


def test_two_weeks_off_steps_back_and_repeats() -> None:
    today = date(2024, 2, 12)
    history = [
        *history_until(date(2024, 1, 22)),
        make_week(date(2024, 1, 29), load=0.0, sessions=0),
        make_week(date(2024, 2, 5), load=0.0, sessions=0),
        make_week(today, load=100.0),
    ]
    rows = by_start(derive_block_plan([make_race(RACE_DAY)], history, today=today))

    assert rows[date(2024, 1, 29)].status == "rest"
    assert rows[date(2024, 2, 5)].status == "rest"
    assert rows[today].status == "step_back"
    assert "Base 1" in rows[today].role
    assert "Z1-2" in rows[today].role
    assert rows[date(2024, 2, 19)].status == "repeat"
    assert rows[date(2024, 4, 15)].status == "omitted"  # Build 2, the first omission


def test_more_than_two_weeks_off_backs_up_one_block() -> None:
    today = date(2024, 2, 12)
    history = [
        *history_until(date(2024, 1, 15)),
        make_week(date(2024, 1, 22), load=0.0, sessions=0),
        make_week(date(2024, 1, 29), load=0.0, sessions=0),
        make_week(date(2024, 2, 5), load=0.0, sessions=0),
        make_week(today, load=100.0),
    ]
    rows = by_start(derive_block_plan([make_race(RACE_DAY)], history, today=today))

    assert rows[date(2024, 1, 22)].status == "rest"
    assert rows[today].status == "step_back"
    assert "Prep" in rows[today].role and "back up one block" in rows[today].role
    assert rows[date(2024, 6, 3)].status == "omitted"  # Peak week 1 comes first
    assert rows[date(2024, 4, 15)].status == "omitted"  # then Build 2
    assert rows[date(2024, 4, 8)].status == "upcoming"  # Build 1 is not omitted


def test_a_build_interruption_steps_back_to_base_3() -> None:
    today = date(2024, 4, 29)
    history = [
        *history_until(date(2024, 4, 8)),
        make_week(date(2024, 4, 15), load=0.0, sessions=0),
        make_week(date(2024, 4, 22), load=0.0, sessions=0),
        make_week(today, load=100.0),
    ]
    rows = by_start(derive_block_plan([make_race(RACE_DAY)], history, today=today))

    assert rows[date(2024, 4, 15)].status == "rest"
    assert rows[today].status == "step_back"
    assert "Base 3" in rows[today].role
    assert rows[date(2024, 5, 6)].status == "repeat"
    assert rows[date(2024, 6, 3)].status == "omitted"


def test_a_one_to_three_day_gap_changes_nothing() -> None:
    today = date(2024, 2, 7)  # Wednesday, two days into the current week
    rows = by_start(
        derive_block_plan([make_race(RACE_DAY)], history_until(date(2024, 1, 29)), today=today)
    )

    assert rows[TODAY].status == "current"
    assert {row.status for row in rows.values()} <= {"done", "under", "over", "current", "upcoming"}


def test_a_session_week_without_load_is_not_a_gap() -> None:
    today = date(2024, 2, 7)  # Wednesday; the last week was strength-only
    history = [
        *history_until(date(2024, 1, 22)),
        make_week(date(2024, 1, 29), load=0.0, sessions=2),
    ]
    rows = by_start(derive_block_plan([make_race(RACE_DAY)], history, today=today))
    statuses = {row.status for row in rows.values()}

    assert statuses.isdisjoint({"rest", "step_back", "repeat", "omitted"})
    assert rows[TODAY].status == "current"


def test_a_late_interruption_changes_nothing() -> None:
    today = date(2024, 6, 10)  # the Peak 2 week
    history = [
        *history_until(date(2024, 5, 20)),
        make_week(date(2024, 5, 27), load=0.0, sessions=0),
        make_week(date(2024, 6, 3), load=0.0, sessions=0),
        make_week(today, load=100.0),
    ]
    rows = derive_block_plan([make_race(RACE_DAY)], history, today=today)
    statuses = {row.status for row in rows}

    assert statuses.isdisjoint({"step_back", "repeat", "omitted"})
    assert by_start(rows)[date(2024, 5, 27)].status == "rest"
    assert by_start(rows)[date(2024, 6, 3)].status == "rest"
    assert by_start(rows)[today].status == "current"


def test_a_deep_interruption_in_the_last_build_week_changes_nothing() -> None:
    today = date(2024, 5, 27)  # Build week 8, the last of the second mesocycle
    history = [
        *history_until(date(2024, 4, 29)),
        make_week(date(2024, 5, 6), load=0.0, sessions=0),
        make_week(date(2024, 5, 13), load=0.0, sessions=0),
        make_week(date(2024, 5, 20), load=0.0, sessions=0),
        make_week(today, load=100.0),
    ]
    rows = derive_block_plan([make_race(RACE_DAY)], history, today=today)
    statuses = {row.status for row in rows}

    assert statuses.isdisjoint({"step_back", "repeat", "omitted"})
    assert by_start(rows)[date(2024, 5, 20)].status == "rest"
    assert by_start(rows)[today].status == "current"


def test_an_illness_steps_back_to_prep() -> None:
    today = date(2024, 2, 12)
    history = [
        *history_until(date(2024, 1, 22)),
        make_week(date(2024, 1, 29), load=0.0, sessions=0),
        make_week(date(2024, 2, 5), load=0.0, sessions=0),
        make_week(today, load=100.0),
    ]
    rows = by_start(derive_block_plan([make_race(RACE_DAY)], history, today=today, illness=True))

    assert rows[today].status == "step_back"
    assert "Prep" in rows[today].role and "illness" in rows[today].role


def test_a_one_week_remainder_joins_the_previous_mesocycle() -> None:
    today = date(2024, 3, 4)
    rows = by_start(
        derive_block_plan([make_race(RACE_DAY)], history_until(today, first=today), today=today)
    )
    assert "Recovery tail" in rows[date(2024, 5, 20)].role  # Build 4 closes a full mesocycle
    assert "Recovery tail" not in rows[date(2024, 5, 27)].role  # Build 5 joins it

    ten_base = by_start(
        derive_block_plan(
            [make_race(RACE_DAY)],
            history_until(today, first=date(2024, 1, 29)),
            today=today,
        )
    )
    assert "Recovery tail" not in ten_base[date(2024, 3, 25)].role  # Base 9
    assert "Recovery tail" in ten_base[date(2024, 4, 1)].role  # Base 10 closes the mesocycle


def test_the_block_plan_renders_as_a_markdown_table_section() -> None:
    rows = derive_block_plan([make_race(RACE_DAY)], history_until(TODAY), today=TODAY)
    context = CoachContext.model_validate(
        {"focus": "plan my race", "goal_races": [make_race(RACE_DAY)], "block_plan": rows}
    )

    table = context.sections()["block_plan"]
    assert table.startswith("| Week | Phase | Role | Target | Status |")
    assert "| 2024-01-01 | Prep 1 |" in table
    assert "| 350-520 | under |" in table
    assert "| 2024-06-17 | Race |" in table
    assert "| 140-350 | upcoming |" in table
    keys = list(context.data_payload())
    assert keys.index("block_plan") == keys.index("goal_races") + 1
    assert context.section_tokens()["block_plan"] > 0


def test_the_block_plan_section_is_absent_when_empty() -> None:
    context = CoachContext.model_validate({"focus": "status check"})

    assert "block_plan" not in context.sections()
    assert context.section_tokens()["block_plan"] == 0


def test_budget_keeps_the_block_plan_over_earlier_sections() -> None:
    race = make_race(RACE_DAY)
    rows = derive_block_plan([race], history_until(TODAY), today=TODAY)
    rollup = history_until(TODAY)
    target = CoachContext(focus="status check", goal_races=[race], block_plan=rows, today=TODAY)
    context = build_within_budget(
        focus="status check",
        recent_activities=[Activity.model_validate(make_activity("fx-old", 1))],
        wellness=[Wellness.model_validate(make_wellness(28))],
        upcoming_events=[],
        sport_settings=[],
        goal_races=[race],
        block_plan=rows,
        training_rollup=rollup,
        user_feedback=None,
        activity_detail=None,
        max_tokens=target.estimated_tokens(),
        today=TODAY,
    )

    assert context.block_plan == rows
    assert context.goal_races == [race]
    assert context.training_rollup == []
    assert context.recent_activities == []
    assert context.wellness == []


def test_budget_evicts_the_block_plan_after_the_goal_races() -> None:
    race = make_race(RACE_DAY)
    rows = derive_block_plan([race], history_until(TODAY), today=TODAY)
    plan_only = CoachContext(focus="status check", block_plan=rows, today=TODAY)
    context = build_within_budget(
        focus="status check",
        recent_activities=[],
        wellness=[],
        upcoming_events=[],
        sport_settings=[],
        goal_races=[race],
        block_plan=rows,
        user_feedback=None,
        activity_detail=None,
        max_tokens=plan_only.estimated_tokens(),
        today=TODAY,
    )

    assert context.block_plan == rows
    assert context.goal_races == []


async def test_gather_context_includes_the_block_plan() -> None:
    client = make_intervals_client(
        events=[
            {
                "id": 7,
                "name": "Summer A",
                "start_date_local": "2024-06-17T00:00:00",
                "category": "RACE_A",
            }
        ]
    )

    context = await gather_context(client, ContextPlan.standard(TODAY), "plan my race", today=TODAY)

    assert [race.name for race in context.goal_races] == ["Summer A"]
    assert context.block_plan[-1].week_start == RACE_DAY
    assert context.block_plan[-1].phase == "Race"
    assert "Target race: Summer A" in context.block_plan[-1].role

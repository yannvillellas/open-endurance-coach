import json
from datetime import date, timedelta
from typing import Any

from open_endurance_coach.config import Settings
from open_endurance_coach.prompts.prompts import (
    DISCUSSION_EXAMPLE,
    INTAKE_EXAMPLE,
    OUTPUT_EXAMPLE,
    RACE_EXAMPLE,
    build_messages,
)
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import (
    CreateRace,
    CreateWorkout,
    DecisionReport,
    DeleteWorkout,
    UpdateRace,
    UpdateWorkout,
)
from open_endurance_coach.tokens import estimate_text_tokens

CONTEXT = CoachContext.model_validate(
    {
        "focus": "Analyze last week's execution",
        "recent_activities": [
            {
                "id": "fx-a",
                "start_date_local": "2024-01-20T08:00:00",
                "type": "Ride",
                "name": "Tempo Session",
                "icu_training_load": 84,
                "icu_average_watts": 240.0,
            }
        ],
        "wellness": [{"id": "2024-01-19", "ctl": 40.0, "hrv": 90.0}],
        "upcoming_events": [{"name": "Long Ride", "start_date_local": "2024-02-10T00:00:00"}],
        "sport_settings": [{"id": 1, "ftp": 250.0}],
        "user_feedback": "Felt tired on Thursday, legs heavy.",
    }
)


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "intervals_api_key": "test-key",
        "deepseek_api_key": "test-llm-key",
        "athlete_profile": "Test athlete: 23-year-old male, criterium racer.",
        "coach_tone": "Be strict and unforgiving.",
    }
    values.update(overrides)
    return Settings(**values)


def test_messages_have_system_and_user_roles() -> None:
    messages = build_messages(CONTEXT, make_settings())
    assert [message.role for message in messages] == ["system", "user"]


def test_output_example_validates_against_the_strict_contract() -> None:
    report = DecisionReport.model_validate(OUTPUT_EXAMPLE)
    assert isinstance(report.mutations[0], CreateWorkout)
    assert isinstance(report.mutations[1], UpdateWorkout)
    assert isinstance(report.mutations[2], DeleteWorkout)


def test_output_example_description_is_documented_workout_text() -> None:
    assert OUTPUT_EXAMPLE["mutations"][0]["description"] == (
        "- 15m 55% Warmup\n\n3x\n- 1m 150%\n- 1m 50%\n\n- 5m 50%\n- 5m 120%\n- 15m 55%"
    )


def test_output_example_repeat_has_blank_line_separation() -> None:
    description = OUTPUT_EXAMPLE["mutations"][0]["description"]
    assert "\n\n3x\n" in description
    assert "1m 50%\n\n- 5m 50%" in description


def test_system_message_instructs_native_workout_text_format() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "native Intervals.icu workout text" in system
    assert "One step per line starting with '- '" in system
    assert "m means minutes" in system


def test_format_rules_carry_official_constructs() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "1m30" in system
    assert "80% (of FTP)" in system
    assert "60% HR (of max heart rate)" in system
    assert "100% LTHR" in system
    assert "Ramp 100-200w" in system
    assert "60m Z2 HR" in system


def test_format_rules_carry_distance_zone_and_pace_forms() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "2.5km Z2 HR" in system
    assert "400mtr Z1 HR" in system
    assert "Z2-Z3 HR" in system
    assert "1:45/100m Pace" in system
    assert "blank line before and after" in system


def test_format_rules_carry_quick_guide_constructs() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "5m30s" in system
    assert "1h2m30s" in system
    assert "1'30\"" in system
    assert "Z2 Pace" in system
    assert "60% MMP 5m" in system
    assert "CZ1" in system
    assert "Nested repeats are not supported" in system
    assert "3:00/100m-4:00/100m Pace" in system


def test_format_rules_carry_step_type_forms() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "on its own line directly above a step" in system
    assert "repeat the label for each step" in system
    assert "- 20m freeride" in system
    assert "- 100mtr Z5 HR MaxEffort" in system
    assert "- 1km ramp 60-50% HR" in system
    assert "bare trailing" in system


def test_system_message_contains_json_word_and_schema_example() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "json" in system.lower()
    example_start = system.find('"summary"')
    assert example_start != -1
    assert "delete" in system[example_start:]


def test_system_message_anchors_dates_to_today() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "never copy the example dates" in system
    assert "on or after today" in system


def test_user_message_states_today_when_known() -> None:
    context = CoachContext.model_validate({**CONTEXT.model_dump(), "today": "2024-02-01"})
    user = build_messages(context, make_settings())[1].content
    assert "Today's date (athlete local)" in user
    assert "2024-02-01" in user


def test_user_message_omits_today_when_unknown() -> None:
    user = build_messages(CONTEXT, make_settings())[1].content
    assert "Today's date" not in user


def test_system_message_contains_coaching_methodology() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "Joe Friel" in system
    assert "periodization" in system
    assert "Coggan" in system


def test_athlete_profile_is_injected_from_settings() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "Test athlete: 23-year-old male, criterium racer." in system


def test_athlete_profile_block_is_omitted_when_unset() -> None:
    system = build_messages(CONTEXT, make_settings(athlete_profile=""))[0].content
    assert "Athlete profile" not in system


def test_coach_tone_is_injected_from_settings() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "Be strict and unforgiving." in system


def test_user_message_contains_focus_and_context_data() -> None:
    user = build_messages(CONTEXT, make_settings())[1].content
    assert "Analyze last week's execution" in user
    assert "fx-a" in user
    assert "2024-02-10" in user
    assert "250.0" in user


def test_user_feedback_is_injected_verbatim_when_present() -> None:
    user = build_messages(CONTEXT, make_settings())[1].content
    assert "Felt tired on Thursday, legs heavy." in user


def test_no_user_feedback_block_without_feedback() -> None:
    context = CoachContext.model_validate({"focus": "status check"})
    user = build_messages(context, make_settings())[1].content
    assert "user feedback" not in user.lower()


def test_build_messages_is_deterministic() -> None:
    settings = make_settings()
    assert build_messages(CONTEXT, settings) == build_messages(CONTEXT, settings)


def test_examples_are_placeholders_not_copyable_answers() -> None:
    for example in (OUTPUT_EXAMPLE, DISCUSSION_EXAMPLE, RACE_EXAMPLE):
        rendered = json.dumps(example)
        assert "2024-" not in rendered
        assert "Tempo Session" not in rendered
        assert "10001" not in rendered
    assert "<workout name>" in json.dumps(OUTPUT_EXAMPLE)
    assert "2099-01-01" in json.dumps(OUTPUT_EXAMPLE)


def test_contract_warns_that_examples_are_shape_only() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "show the shape only" in system
    assert "a mutation that sets a date before today is rejected" in system


def test_contract_defaults_to_discussion_policy() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert 'Return an empty mutations list unless intent is "plan"' in system
    assert "conversation - no calendar change requested" in system
    assert "asked for a plan or calendar change" in system


def test_examples_declare_intent() -> None:
    assert DISCUSSION_EXAMPLE["intent"] == "chat"
    assert DISCUSSION_EXAMPLE["mutations"] == []
    assert OUTPUT_EXAMPLE["intent"] == "plan"
    assert OUTPUT_EXAMPLE["mutations"]
    assert DecisionReport(summary="x").intent == "analysis"


def test_contract_blocks_proposals_on_material_questions() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "Before prescribing anything" in system
    assert "Put those questions in needs_input" in system
    assert "never assume on the athlete's behalf" in system
    assert "Assume only when the athlete explicitly tells you to" in system
    assert "state the assumption in the summary" in system
    assert "Never list the same question in both questions and needs_input" in system


def test_contract_requires_topic_labels_on_findings() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "Every finding must start with a short topic label" in system


def test_contract_tells_the_model_to_compare_recent_events() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "recent_events lists the calendar entries" in system
    assert "compare it against recent_activities" in system


def test_contract_allows_distance_on_workout_mutations() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "Workout mutations may also set distance" in system


def test_contract_forbids_distance_steps_without_a_pace_model() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "No pace model for Hike or Walk" in system
    assert "no distance steps" in system


def test_intake_example_demonstrates_blocking() -> None:
    report = DecisionReport.model_validate(INTAKE_EXAMPLE)
    assert report.intent == "plan"
    assert report.needs_input
    assert report.mutations == []


def test_examples_declare_needs_input() -> None:
    assert OUTPUT_EXAMPLE["needs_input"] == []
    assert DISCUSSION_EXAMPLE["needs_input"] == []
    assert RACE_EXAMPLE["needs_input"] == []
    assert INTAKE_EXAMPLE["needs_input"]


def test_full_rollup_context_fits_default_budget() -> None:
    rollup = [
        {
            "week_start": (date(2024, 1, 5) + timedelta(days=7 * index)).isoformat(),
            "partial": index == 13,
            "sessions": 5,
            "time_s": 14400,
            "load": 320.0,
            "fitness": 45.2,
            "fatigue": 38.1,
            "form": 7.1,
            "ramp_rate": 1.8,
            "sports": [
                {"category": "Ride", "sessions": 3, "time_s": 9000, "load": 210.0},
                {"category": "Run", "sessions": 2, "time_s": 5400, "load": 110.0},
            ],
        }
        for index in range(14)
    ]
    context = CoachContext.model_validate({"focus": "plan my race", "training_rollup": rollup})
    assert context.estimated_tokens() <= 4096


def test_race_example_validates_and_is_placeholder_only() -> None:
    report = DecisionReport.model_validate(RACE_EXAMPLE)
    assert report.intent == "plan"
    assert isinstance(report.mutations[0], CreateRace)
    assert isinstance(report.mutations[1], UpdateRace)
    assert RACE_EXAMPLE["mutations"][0]["category"] == "RACE_B"
    assert RACE_EXAMPLE["mutations"][0]["moving_time"] == 0
    assert RACE_EXAMPLE["mutations"][0]["icu_training_load"] == 0
    assert "<race name>" in json.dumps(RACE_EXAMPLE)
    assert RACE_EXAMPLE["mutations"][1]["moving_time"] == 0
    assert RACE_EXAMPLE["mutations"][1]["icu_training_load"] == 0
    assert RACE_EXAMPLE["mutations"][0]["distance"] == 0


def test_contract_teaches_backwards_planning_and_asking() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "plan backwards from the nearest race" in system
    assert "Base, Build, Peak, Taper" in system
    assert "next 7-14 days only" in system
    assert "RACE_A (season objective), RACE_B (important) or RACE_C" in system
    assert "ask the athlete for it instead of estimating" in system
    assert "Never copy the example race numbers" in system
    assert "When a race is missing a figure" in system
    assert "TrailRun" in system


def test_system_prompt_stays_small() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert estimate_text_tokens(system) <= 2560


def test_contract_forbids_no_op_race_updates() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "Never re-propose a race mutation whose fields already match goal_races" in system

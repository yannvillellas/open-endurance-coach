from typing import Any

from open_endurance_coach.config import Settings
from open_endurance_coach.prompts.prompts import OUTPUT_EXAMPLE, build_messages
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import (
    CreateWorkout,
    DecisionReport,
    DeleteWorkout,
    UpdateWorkout,
)

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


def test_system_message_contains_json_word_and_schema_example() -> None:
    system = build_messages(CONTEXT, make_settings())[0].content
    assert "json" in system.lower()
    example_start = system.find('"summary"')
    assert example_start != -1
    assert "delete" in system[example_start:]


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

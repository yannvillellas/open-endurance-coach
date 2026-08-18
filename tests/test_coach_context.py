import pytest
from pydantic import ValidationError

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.intervals import Activity, Wellness

ACTIVITY = {
    "id": "fx000001",
    "start_date_local": "2024-01-10T08:00:00",
    "type": "Ride",
    "name": "Tempo Session",
    "icu_training_load": 84,
    "moving_time": 3600,
    "icu_average_watts": 245.0,
    "average_heartrate": 152.0,
}

WELLNESS = {"id": "2024-01-10", "ctl": 45.0, "hrv": 95.0, "sleepSecs": 28000}


def make_activity(**overrides: object) -> dict[str, object]:
    return {**ACTIVITY, **overrides}


def test_valid_context_with_all_sections_constructs() -> None:
    activity = Activity.model_validate(ACTIVITY)
    wellness = Wellness.model_validate(WELLNESS)
    context = CoachContext(
        focus="Analyze last week's execution",
        recent_activities=[activity],
        wellness=[wellness],
        sport_settings=[],
        upcoming_events=[],
        user_feedback="Felt tired on Thursday",
    )
    assert context.focus == "Analyze last week's execution"
    assert context.recent_activities == [activity]
    assert context.user_feedback == "Felt tired on Thursday"
    assert context.activity_detail is None


def test_focus_is_required_and_non_empty() -> None:
    with pytest.raises(ValidationError):
        CoachContext.model_validate({})
    with pytest.raises(ValidationError):
        CoachContext.model_validate({"focus": ""})


def test_sections_default_to_empty() -> None:
    context = CoachContext.model_validate({"focus": "status check"})
    assert context.recent_activities == []
    assert context.wellness == []
    assert context.upcoming_events == []
    assert context.sport_settings == []
    assert context.activity_detail is None
    assert context.user_feedback is None


def test_estimated_tokens_grow_with_content() -> None:
    small = CoachContext.model_validate({"focus": "status check"})
    larger = CoachContext.model_validate(
        {
            "focus": "status check",
            "recent_activities": [ACTIVITY, ACTIVITY, ACTIVITY],
        }
    )
    assert larger.estimated_tokens() > small.estimated_tokens()


def test_context_over_budget_is_rejected() -> None:
    with pytest.raises(ValidationError, match="token budget"):
        CoachContext.model_validate(
            {"focus": "status check", "recent_activities": [ACTIVITY], "max_tokens": 20}
        )
    CoachContext.model_validate(
        {"focus": "status check", "recent_activities": [ACTIVITY], "max_tokens": 4000}
    )


def test_section_tokens_reports_per_section() -> None:
    context = CoachContext.model_validate(
        {"focus": "status check", "recent_activities": [ACTIVITY], "wellness": [WELLNESS]}
    )
    sections = context.section_tokens()
    assert set(sections) == {
        "focus",
        "today",
        "recent_activities",
        "activity_detail",
        "wellness",
        "upcoming_events",
        "sport_settings",
        "user_feedback",
    }
    assert sections["recent_activities"] > 0
    assert sections["wellness"] > 0
    assert sections["activity_detail"] == 0
    assert sections["user_feedback"] == 0
    assert sections["focus"] > 0


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        CoachContext.model_validate({"focus": "x", "max_tokens": 0})


def test_unknown_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CoachContext.model_validate({"focus": "x", "hallucinated": 1})

import json

import pytest
from pydantic import ValidationError

from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.intervals import Activity, ActivitySplit, Wellness
from open_endurance_coach.tokens import (
    CHARS_PER_TOKEN,
    estimate_payload_tokens,
    estimate_text_tokens,
)

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
    assert context.recent_events == []
    assert context.sport_settings == []
    assert context.activity_detail is None
    assert context.activity_splits == []
    assert context.user_feedback is None


def test_estimated_tokens_match_the_indented_payload() -> None:
    context = CoachContext.model_validate(
        {"focus": "status check", "recent_activities": [ACTIVITY], "max_tokens": 4096}
    )
    data = context.data_payload()
    compact = len(json.dumps(data, ensure_ascii=False)) // CHARS_PER_TOKEN
    indented = estimate_payload_tokens(data)
    assert indented > compact
    assert context.data_tokens() == indented
    assert context.estimated_tokens() == indented + estimate_text_tokens(context.focus)


@pytest.mark.parametrize("section", ["recent_events", "upcoming_events"])
def test_event_sections_drop_machine_fields_but_keep_the_rest(section: str) -> None:
    event = {
        "id": 136743074,
        "name": "Hill Sharpening",
        "start_date_local": "2024-01-31T00:00:00",
        "category": "WORKOUT",
        "type": "TrailRun",
        "description": "- 10m Z1 warmup",
        "end_date_local": "2024-01-31T00:00:00",
        "moving_time": 1800,
        "distance": 5000.0,
        "icu_training_load": 36.0,
        "workout_doc": {"steps": [{"duration": 600}]},
        "plan_folder_id": 42,
        "plan_workout_id": 43,
    }
    context = CoachContext.model_validate(
        {"focus": "status check", section: [event], "max_tokens": 4096}
    )
    payload = context.sections()[section][0]
    assert payload["id"] == 136743074
    assert payload["description"] == "- 10m Z1 warmup"
    assert payload["end_date_local"] == "2024-01-31T00:00:00"
    assert payload["moving_time"] == 1800
    assert payload["distance"] == 5000.0
    assert payload["icu_training_load"] == 36.0
    for dropped in ("workout_doc", "plan_folder_id", "plan_workout_id"):
        assert dropped not in payload


def test_prompt_includes_past_planned_events() -> None:
    from open_endurance_coach.config import Settings
    from open_endurance_coach.prompts.prompts import build_messages

    context = CoachContext.model_validate(
        {
            "focus": "review yesterday",
            "today": "2026-09-19",
            "recent_events": [
                {
                    "id": 136743074,
                    "name": "Hill Sharpening 3x3m Z4-Z5 HR",
                    "start_date_local": "2026-09-18T00:00:00",
                    "category": "WORKOUT",
                    "type": "TrailRun",
                }
            ],
        }
    )
    settings = Settings(intervals_api_key="k", deepseek_api_key="k")
    prompt = build_messages(context, settings)[1].content
    assert "Hill Sharpening 3x3m Z4-Z5 HR" in prompt


def test_sections_match_the_prompt_payload() -> None:
    from open_endurance_coach.config import Settings
    from open_endurance_coach.prompts.prompts import build_messages

    context = CoachContext.model_validate(
        {
            "focus": "status check",
            "today": "2024-02-01",
            "recent_activities": [ACTIVITY],
            "user_feedback": "legs heavy",
            "max_tokens": 4096,
        }
    )
    settings = Settings(intervals_api_key="k", deepseek_api_key="k")
    user = build_messages(context, settings)[1].content
    start = user.index("<athlete_data>\n") + len("<athlete_data>\n")
    end = user.index("\n</athlete_data>")
    payload = json.loads(user[start:end])
    expected = {key: value for key, value in context.sections().items() if key != "focus"}
    assert payload == expected
    assert "Current message:\nstatus check" in user
    assert "Today's date (athlete local): 2024-02-01" in user
    assert "<athlete_data>" in user and "</athlete_data>" in user


def test_estimated_tokens_grow_with_content() -> None:
    small = CoachContext.model_validate({"focus": "status check"})
    larger = CoachContext.model_validate(
        {
            "focus": "status check",
            "recent_activities": [ACTIVITY, ACTIVITY, ACTIVITY],
        }
    )
    assert larger.estimated_tokens() > small.estimated_tokens()


def test_activity_splits_survive_a_json_round_trip() -> None:
    context = CoachContext(
        focus="status check",
        activity_splits=[ActivitySplit(label="km 1-2 (partial)", distance_m=1500.0, time_s=300)],
    )
    restored = CoachContext.model_validate(context.model_dump(mode="json"))
    assert restored.activity_splits == context.activity_splits


def test_context_over_budget_is_rejected() -> None:
    with pytest.raises(ValidationError, match="token budget"):
        CoachContext.model_validate(
            {"focus": "status check", "recent_activities": [ACTIVITY], "max_tokens": 20}
        )
    CoachContext.model_validate(
        {"focus": "status check", "recent_activities": [ACTIVITY], "max_tokens": 4000}
    )


def test_section_tokens_order_matches_the_rendered_sections() -> None:
    context = CoachContext.model_validate(
        {
            "focus": "status check",
            "recent_activities": [ACTIVITY],
            "activity_splits": [{"label": "km 1", "distance_m": 1000.0, "time_s": 300}],
            "today": "2024-02-01",
            "user_feedback": "felt tired",
            "current_proposal": {
                "intent": "chat",
                "summary": "hello",
                "findings": [],
                "questions": [],
                "needs_input": [],
                "mutations": [],
            },
        }
    )
    sections = context.section_tokens()
    rendered = [key for key in context.sections() if key in sections]
    assert rendered == [key for key in sections if key in rendered]
    assert {"activity_splits", "today", "current_proposal", "user_feedback"} <= set(rendered)


def test_section_tokens_reports_per_section() -> None:
    context = CoachContext.model_validate(
        {"focus": "status check", "recent_activities": [ACTIVITY], "wellness": [WELLNESS]}
    )
    sections = context.section_tokens()
    rendered = [key for key in context.sections() if key in sections]
    assert rendered == [key for key in sections if key in rendered]
    assert set(sections) == {
        "focus",
        "today",
        "current_proposal",
        "recent_activities",
        "activity_detail",
        "activity_splits",
        "wellness",
        "recent_events",
        "upcoming_events",
        "goal_races",
        "training_rollup",
        "sport_settings",
        "user_feedback",
    }
    assert sections["recent_activities"] > 0
    assert sections["wellness"] > 0
    assert sections["activity_detail"] == 0
    assert sections["user_feedback"] == 0
    assert sections["focus"] > 0


def test_empty_sections_cost_only_the_payload_they_render() -> None:
    context = CoachContext.model_validate({"focus": "status check"})
    tokens = context.section_tokens()
    assert tokens["recent_activities"] == 0
    assert tokens["wellness"] == 0
    assert tokens["user_feedback"] == 0
    assert context.data_tokens() == estimate_payload_tokens(context.data_payload())


def test_focus_is_bounded_by_the_request_not_the_data_budget() -> None:
    context = CoachContext.model_validate({"focus": "x" * 60000, "recent_activities": [ACTIVITY]})
    assert context.data_tokens() <= context.max_tokens
    assert context.estimated_tokens() > context.max_tokens


def test_max_tokens_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        CoachContext.model_validate({"focus": "x", "max_tokens": 0})


def test_unknown_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CoachContext.model_validate({"focus": "x", "hallucinated": 1})


def test_goal_races_serialize_into_sections() -> None:
    context = CoachContext.model_validate(
        {
            "focus": "status check",
            "goal_races": [
                {
                    "event_id": 90001,
                    "name": "Spring Half",
                    "date": "2024-03-01",
                    "category": "RACE_A",
                    "type": "Run",
                    "days_to_race": 29,
                    "weeks_to_race": 5,
                    "phase": "Build",
                    "distance": 21097.5,
                }
            ],
        }
    )
    assert context.sections()["goal_races"] == [
        {
            "event_id": 90001,
            "name": "Spring Half",
            "date": "2024-03-01",
            "category": "RACE_A",
            "type": "Run",
            "days_to_race": 29,
            "weeks_to_race": 5,
            "phase": "Build",
            "distance": 21097.5,
        }
    ]
    assert context.section_tokens()["goal_races"] > 0


def test_goal_race_rejects_unknown_category_and_extras() -> None:
    base = {
        "name": "Spring Half",
        "date": "2024-03-01",
        "category": "RACE_A",
        "days_to_race": 29,
        "weeks_to_race": 5,
        "phase": "Build",
    }
    with pytest.raises(ValidationError):
        CoachContext.model_validate({"focus": "x", "goal_races": [{**base, "category": "WORKOUT"}]})
    with pytest.raises(ValidationError):
        CoachContext.model_validate({"focus": "x", "goal_races": [{**base, "phase": "Recovery"}]})
    with pytest.raises(ValidationError):
        CoachContext.model_validate({"focus": "x", "goal_races": [{**base, "hallucinated": True}]})

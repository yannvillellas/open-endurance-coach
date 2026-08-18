import json
from typing import Any

from open_endurance_coach.clients.llm import LlmMessage
from open_endurance_coach.config import Settings
from open_endurance_coach.schemas.context import CoachContext

OUTPUT_EXAMPLE: dict[str, Any] = {
    "summary": "Execution matched targets; keep load stable.",
    "findings": ["Thursday's tempo block executed 8% above target power."],
    "questions": ["What was your RPE on Thursday's session?"],
    "mutations": [
        {
            "action": "create",
            "name": "Tempo Session",
            "start_date_local": "2024-01-05",
            "description": "3x10min sweet spot",
            "type": "Ride",
            "moving_time": 3600,
            "icu_training_load": 84,
        },
        {"action": "update", "event_id": 10001, "moving_time": 4200},
        {"action": "delete", "event_id": 10002},
    ],
}

METHODOLOGY = (
    "You are an elite endurance coach enforcing Joe Friel's periodization principles "
    "and Dr. Andrew Coggan's power-based analytics.\n"
    "You validate executed training against planned targets, contextualize with the "
    "athlete's subjective feedback, and modulate upcoming load accordingly.\n"
)


def _json_contract() -> str:
    return (
        "Respond with a single json object and nothing else, matching this exact "
        "schema. The word json in this instruction is required for strict JSON mode.\n"
        "Every start_date_local must be on or after today (the athlete's local date), "
        "taken from the upcoming schedule - never copy the example dates.\n"
        "Example json:\n"
        f"{json.dumps(OUTPUT_EXAMPLE, indent=2)}\n"
    )


def _system_message(settings: Settings) -> str:
    parts = [METHODOLOGY, settings.coach_tone + "\n"]
    if settings.athlete_profile:
        parts.append(f"Athlete profile: {settings.athlete_profile}\n")
    parts.append(_json_contract())
    return "".join(parts)


def _user_message(context: CoachContext) -> str:
    sections: dict[str, Any] = {
        "focus": context.focus,
        "recent_activities": [
            item.model_dump(mode="json", exclude_none=True) for item in context.recent_activities
        ],
        "activity_detail": (
            context.activity_detail.model_dump(mode="json", exclude_none=True)
            if context.activity_detail
            else None
        ),
        "wellness": [item.model_dump(mode="json", exclude_none=True) for item in context.wellness],
        "upcoming_events": [
            item.model_dump(mode="json", exclude_none=True) for item in context.upcoming_events
        ],
        "sport_settings": [
            item.model_dump(mode="json", exclude_none=True) for item in context.sport_settings
        ],
    }
    if context.today:
        sections["today"] = f"Today's date (athlete local): {context.today.isoformat()}"
    if context.user_feedback:
        sections["user_feedback"] = context.user_feedback
    return (
        "Athlete data:\n"
        f"{json.dumps(sections, indent=2, ensure_ascii=False)}\n"
        "Produce your analysis as json per the contract.\n"
    )


def build_messages(context: CoachContext, settings: Settings) -> list[LlmMessage]:
    return [
        LlmMessage(role="system", content=_system_message(settings)),
        LlmMessage(role="user", content=_user_message(context)),
    ]

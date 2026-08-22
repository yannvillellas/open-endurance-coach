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
            "description": (
                "- 15m 55% Warmup\n\n3x\n- 1m 150%\n- 1m 50%\n\n- 5m 50%\n- 5m 120%\n- 15m 55%"
            ),
            "type": "Ride",
            "moving_time": 3600,
            "icu_training_load": 84,
        },
        {"action": "update", "event_id": 10001, "moving_time": 4200},
        {"action": "delete", "event_id": 10002},
    ],
}

# Native Intervals.icu workout text, as documented by the Intervals.icu workout builder
# (forum topic 1163), the workout builder syntax quick guide (123701), distance-based
# workouts (9973) and absolute pace (115846). Only the Warmup/Cooldown label lines, the
# MaxEffort keyword and the bare-ramp warning are not in the official docs - they were
# confirmed by real API write/read-back tests on 2026-08-22.
WORKOUT_TEXT_FORMAT = (
    "Workout descriptions must use the native Intervals.icu workout text format.\n"
    "One step per line starting with '- '.\n"
    "Durations: 30s, 10m, 1m30, 5m30s, 1h2m30s (h hours, m minutes, s seconds;"
    " short forms 5', 30\", 1'30\").\n"
    "Targets: 100w, 80% (of FTP), 60% HR (of max heart rate), 100% LTHR"
    " (of threshold HR), 90rpm (cadence).\n"
    "Ranges: 100-140w, 80-90%. Ramps: Ramp 100-200w, Ramp 60-80%.\n"
    "Zones: - 60m Z2 (power zone), - 60m Z2 HR (heart rate zone), - 10m Z2 Pace;"
    " zone ranges work too (Z2-Z3 HR, Z3-Z4).\n"
    "MMP targets: 60% MMP 5m. Custom zones: CZ1, CZ2-CZ3.\n"
    "Distance steps: - 2.5km Z2 HR, - 400mtr Z1 HR (units km, mi, mtr - never"
    " plain m, m means minutes).\n"
    "Pace steps: - 10m 7:15-7:00/km Pace, - 0.1km 1:45/100m Pace (units /km /mi"
    " /100m /500m /250m /400m /100y; ranges like 3:00/100m-4:00/100m Pace work).\n"
    "Repeats: put 4x (or Main set 4x) on its own line before the repeated steps,"
    " no '- ' prefix, with a blank line before and after the block. Nested repeats"
    " are not supported.\n"
    "Step types: 'Warmup' or 'Cooldown' on its own line directly above a step"
    " marks that one step (repeat the label for each step). 'freeride' and"
    " 'MaxEffort' are keywords inside the step line (- 20m freeride, - 100mtr Z5"
    " HR MaxEffort). 'ramp' needs a range (- 1km ramp 60-50% HR); a bare trailing"
    " 'ramp' does not create a ramp step.\n"
    "Text before the number on a step line becomes the step prompt (Recovery 30s"
    " 50%). Other prose lines are ignored for steps.\n"
)

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
        "If current_proposal is present in the athlete data, revise that proposal "
        "minimally to satisfy the user feedback - do not redesign from scratch.\n"
        "Copy the workout text format from the example, not the example's numbers.\n"
        f"{WORKOUT_TEXT_FORMAT}"
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
    return (
        "Athlete data:\n"
        f"{json.dumps(context.sections(), indent=2, ensure_ascii=False)}\n"
        "Produce your analysis as json per the contract.\n"
    )


def build_messages(context: CoachContext, settings: Settings) -> list[LlmMessage]:
    return [
        LlmMessage(role="system", content=_system_message(settings)),
        LlmMessage(role="user", content=_user_message(context)),
    ]

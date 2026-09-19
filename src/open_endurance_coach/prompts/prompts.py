import json
from typing import Any

from open_endurance_coach.clients.llm import LlmMessage
from open_endurance_coach.config import Settings
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.tokens import estimate_text_tokens

OUTPUT_EXAMPLE: dict[str, Any] = {
    "intent": "plan",
    "summary": "<one-line summary of the athlete's data>",
    "findings": ["<topic label>: <finding grounded in the athlete's data>"],
    "questions": ["<optional question for the athlete>"],
    "needs_input": [],
    "mutations": [
        {
            "action": "create",
            "name": "<workout name>",
            "start_date_local": "2099-01-01",
            "description": (
                "- 15m 55% Warmup\n\n3x\n- 1m 150%\n- 1m 50%\n\n- 5m 50%\n- 5m 120%\n- 15m 55%"
            ),
            "type": "<Run | Ride | Swim | ...>",
            "moving_time": 0,
            "distance": 0,
            "icu_training_load": 0,
        },
        {"action": "update", "event_id": "EVENT_ID", "moving_time": 0},
        {"action": "delete", "event_id": "EVENT_ID"},
    ],
}

DISCUSSION_EXAMPLE: dict[str, Any] = {
    "intent": "chat",
    "summary": "<direct answer to the athlete's question>",
    "findings": ["<topic label>: <supporting observation from the athlete's data>"],
    "questions": ["<follow-up question for the athlete>"],
    "needs_input": [],
    "mutations": [],
}

INTAKE_EXAMPLE: dict[str, Any] = {
    "intent": "plan",
    "summary": "<what you can already say, and the assumption you must not make>",
    "findings": ["<topic label>: <what the data shows>"],
    "questions": [],
    "needs_input": ["<the fact you need before a plan is possible>"],
    "mutations": [],
}

RACE_EXAMPLE: dict[str, Any] = {
    "intent": "plan",
    "summary": "<macro outline for the race countdown: Base, Build, Peak, Taper>",
    "findings": [
        "<topic label>: <what the weekly rollup and readiness say about the current phase>"
    ],
    "questions": ["<optional question for the athlete>"],
    "needs_input": [],
    "mutations": [
        {
            "action": "create_race",
            "name": "<race name>",
            "start_date_local": "2099-01-01",
            "category": "RACE_B",
            "type": "<Run | TrailRun | Ride | Swim | Hike | ...>",
            "moving_time": 0,
            "distance": 0,
            "icu_training_load": 0,
        },
        {
            "action": "update_race",
            "event_id": "EVENT_ID",
            "category": "RACE_A",
            "moving_time": 0,
            "icu_training_load": 0,
        },
    ],
}

PROPOSAL_POLICY = (
    'Classify the athlete\'s request in intent: "chat" for questions, advice, '
    'explanation or discussion; "analysis" for a review of executed training; '
    '"plan" only when the athlete asks for a calendar change or a training plan '
    "(plan, schedule, create, add, adjust, taper, reschedule). Return an empty "
    'mutations list unless intent is "plan": for chat and analysis put the answer in '
    "summary/findings and, if a change would help, offer it as a question - never "
    "encode a change the athlete did not ask for.\n"
    "recent_events lists the calendar entries from the last 14 days (workouts, races "
    "and notes) and may be trimmed under budget pressure: compare it against "
    "recent_activities before judging whether the athlete followed the plan, and say "
    "which prescribed session each executed activity does or does not match.\n"
    "Every finding must start with a short topic label of one to three words followed "
    "by ': ' so the list can be scanned (for example 'Wellness: ...', 'Load: ...'); "
    "keep the full detail after the label, never shorten the finding.\n"
    "The examples below show the shape only: every field must come from the athlete "
    "data, the example workout text is the only thing to imitate, never copy the "
    "example dates, and any start_date_local must be on or after today from the "
    "upcoming schedule, since a mutation that sets a date before today is rejected.\n"
    "When goal_races is present, plan backwards from the nearest race: state the macro "
    "phases (Base, Build, Peak, Taper) with weekly load targets in the summary, then "
    "propose concrete workouts for the next 7-14 days only - do not schedule sessions "
    "beyond the visible calendar window.\n"
    "Race events use category RACE_A (season objective), RACE_B (important) or RACE_C "
    "(training race). If a distance, elevation gain or expected load is missing from a "
    "race, ask the athlete for it instead of estimating. When a race is missing a "
    "figure, set moving_time (planned duration), distance (metres) and "
    "icu_training_load from their figures or the authorised target, so no plan is "
    "built on an unknown race load. Never "
    "re-propose a race mutation whose fields already match goal_races: mutate a race "
    "only when a value is missing or must change. Use the canonical sport as type - Run, TrailRun "
    "(trail races), VirtualRun, Ride, VirtualRide, GravelRide, MountainBikeRide, Swim, "
    "Hike, Other - never the example placeholder. Never copy the example race "
    "numbers.\n"
    "Workout mutations may also set distance in metres.\n"
    "Hike or Walk have no pace model: no distance steps, and rewrite an existing "
    "distance step when revising one; use name, distance and a plain description or "
    "time step.\n"
    "Before prescribing anything - workouts, a race, or a full block - list what you "
    "still need to know that would change the plan "
    "(athlete goals, available days, constraints, injury, RPE; race duration, elevation, "
    "expected load). Put those questions in needs_input and, when it is non-empty, return "
    "no mutations and ask - never assume on the athlete's behalf, even when the data lets "
    "you estimate. Assume only when the athlete explicitly tells you to: then state the "
    "assumption in the summary and plan. Never list the same question in both questions "
    "and needs_input.\n"
)

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
    "Distance steps (not Hike or Walk): - 2.5km Z2 HR, - 400mtr Z1 HR (units km, mi, mtr"
    " - never plain m, m means minutes).\n"
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
        f"{PROPOSAL_POLICY}"
        "If current_proposal is present in the athlete data, revise that proposal "
        "minimally to satisfy the user feedback - do not redesign from scratch.\n"
        f"{WORKOUT_TEXT_FORMAT}"
        "Example json (material information missing - ask, do not plan):\n"
        f"{json.dumps(INTAKE_EXAMPLE, indent=2)}\n"
        "Example json (conversation - no calendar change requested):\n"
        f"{json.dumps(DISCUSSION_EXAMPLE, indent=2)}\n"
        "Example json (the athlete asked for a plan or calendar change):\n"
        f"{json.dumps(OUTPUT_EXAMPLE, indent=2)}\n"
        "Example json (a goal race is in the athlete data - plan backwards from it):\n"
        f"{json.dumps(RACE_EXAMPLE, indent=2)}\n"
    )


def system_prompt(settings: Settings) -> str:
    return _system_message(settings)


def _system_message(settings: Settings) -> str:
    parts = [METHODOLOGY, settings.coach_tone + "\n"]
    if settings.athlete_profile:
        parts.append(f"Athlete profile: {settings.athlete_profile}\n")
    parts.append(_json_contract())
    return "".join(parts)


def _user_message(context: CoachContext, history: list[LlmMessage] | None = None) -> str:
    data = context.sections()
    focus = str(data.pop("focus", ""))
    parts = [f"Athlete data:\n{json.dumps(data, indent=2, ensure_ascii=False)}\n"]
    if history:
        transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in history)
        parts.append(f"Recent conversation:\n{transcript}\n")
    parts.append(f"Current message:\n{focus}\n")
    parts.append("Respond per the contract.\n")
    return "".join(parts)


def estimate_user_message_tokens(
    context: CoachContext, history: list[LlmMessage] | None = None
) -> int:
    """Tokens of the exact user message the prompt builder will send."""
    return estimate_text_tokens(_user_message(context, history))


def build_messages(
    context: CoachContext,
    settings: Settings,
    history: list[LlmMessage] | None = None,
) -> list[LlmMessage]:
    return [
        LlmMessage(role="system", content=_system_message(settings)),
        LlmMessage(role="user", content=_user_message(context, history)),
    ]

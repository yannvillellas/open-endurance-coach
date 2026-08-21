import json

from open_endurance_coach.clients.llm import LlmMessage
from open_endurance_coach.config import Settings
from open_endurance_coach.prompts.prompts import METHODOLOGY, context_sections
from open_endurance_coach.schemas.context import CoachContext

CHAT_DIRECTIVE = (
    "You are in a continuing conversation with the athlete.\n"
    "Answer directly and concisely, grounding every claim in the data provided."
    " Do not fabricate numbers.\n"
    "Calendar changes are never made in conversation: propose them in prose and point to"
    " /analyze, then the athlete approves or rejects through the review flow with explicit"
    " confirmation. Formal answers to your earlier questions go through /feedback.\n"
)


def _chat_system(settings: Settings) -> str:
    parts = [METHODOLOGY, settings.coach_tone + "\n"]
    if settings.athlete_profile:
        parts.append(f"Athlete profile: {settings.athlete_profile}\n")
    parts.append(CHAT_DIRECTIVE)
    return "".join(parts)


def build_chat_messages(
    context: CoachContext,
    settings: Settings,
    *,
    history: list[LlmMessage] | None = None,
    text: str,
) -> list[LlmMessage]:
    messages = [
        LlmMessage(role="system", content=_chat_system(settings)),
        LlmMessage(
            role="user",
            content=(
                "Athlete data:\n"
                + json.dumps(context_sections(context), indent=2, ensure_ascii=False)
            ),
        ),
    ]
    messages.extend(history or [])
    messages.append(LlmMessage(role="user", content=text))
    return messages

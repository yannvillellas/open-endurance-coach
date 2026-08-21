from datetime import date

from open_endurance_coach.clients.llm import LlmMessage
from open_endurance_coach.config import Settings
from open_endurance_coach.prompts.chat import build_chat_messages
from open_endurance_coach.schemas.context import CoachContext


def test_chat_messages_shape(settings: Settings) -> None:
    history = [
        LlmMessage(role="user", content="RPE was 7"),
        LlmMessage(role="assistant", content="Noted."),
    ]
    messages = build_chat_messages(
        CoachContext(focus="how was my week"), settings, history=history, text="and today?"
    )
    assert [message.role for message in messages] == ["system", "user", "user", "assistant", "user"]
    assert messages[-1].content == "and today?"


def test_chat_system_carries_persona_tone_and_profile(settings: Settings) -> None:
    personalized = settings.model_copy(
        update={"coach_tone": "Be strict.", "athlete_profile": "23yo criterium racer"}
    )
    messages = build_chat_messages(CoachContext(focus="f"), personalized, text="hi")
    system = messages[0].content
    assert "Joe Friel's periodization" in system
    assert "Be strict." in system
    assert "Athlete profile: 23yo criterium racer" in system


def test_chat_system_omits_profile_when_empty(settings: Settings) -> None:
    messages = build_chat_messages(
        CoachContext(focus="f"), settings.model_copy(update={"athlete_profile": ""}), text="hi"
    )
    assert "Athlete profile" not in messages[0].content


def test_chat_system_has_no_json_contract(settings: Settings) -> None:
    messages = build_chat_messages(CoachContext(focus="f"), settings, text="hi")
    system = messages[0].content
    assert "json" not in system.casefold()


def test_chat_data_message_contains_context_sections(settings: Settings) -> None:
    context = CoachContext(
        focus="how was my week",
        today=date(2024, 2, 1),
        user_feedback="legs tired",
    )
    messages = build_chat_messages(context, settings, text="hi")
    data = messages[1].content
    assert "Athlete data:" in data
    assert "how was my week" in data
    assert "2024-02-01" in data
    assert "legs tired" in data


def test_chat_history_sits_between_data_and_turn(settings: Settings) -> None:
    history = [LlmMessage(role="user", content="past question")]
    messages = build_chat_messages(CoachContext(focus="f"), settings, history=history, text="now")
    assert messages[2] == history[0]


def test_chat_history_is_optional(settings: Settings) -> None:
    messages = build_chat_messages(CoachContext(focus="f"), settings, text="hi")
    assert [message.role for message in messages] == ["system", "user", "user"]

from datetime import UTC, datetime

import pytest

from open_endurance_coach.chat.history import (
    ChatSession,
    assistant_turn,
    seed_turns,
    trim_history,
)
from open_endurance_coach.clients.llm import LlmMessage
from open_endurance_coach.schemas.decisions import DecisionReport
from open_endurance_coach.store.records import Message, MessageRole
from open_endurance_coach.tokens import CHARS_PER_TOKEN, estimate_text_tokens

NOW = datetime(2024, 2, 1, 12, 0, 0, tzinfo=UTC)


def message(message_id: int, role: MessageRole, content: str) -> Message:
    return Message(id=message_id, created_at=NOW, role=role, content=content)


def test_seed_turns_empty() -> None:
    assert seed_turns([]) == []


def test_seed_turns_replays_messages_oldest_first() -> None:
    entries = [
        message(2, MessageRole.ASSISTANT, "Load stable."),
        message(1, MessageRole.USER, "plan my week"),
    ]
    assert seed_turns(entries) == [
        LlmMessage(role="user", content="plan my week"),
        LlmMessage(role="assistant", content="Load stable."),
    ]


def test_seed_turns_keeps_a_user_row_without_a_reply() -> None:
    turns = seed_turns([message(1, MessageRole.USER, "orphan")])
    assert turns == [LlmMessage(role="user", content="orphan")]


def test_assistant_turn_is_the_summary_with_findings() -> None:
    report = DecisionReport(summary="Summary.", findings=["Finding A1.", "Finding A2."])
    assert assistant_turn(report).content == "Summary.\n- Finding A1.\n- Finding A2."


def test_assistant_turn_replaces_unpaired_surrogates() -> None:
    report = DecisionReport(summary="Summary.", findings=["finding \udfff"])
    content = assistant_turn(report).content
    assert "\udfff" not in content
    assert "\ufffd" in content
    content.encode("utf-8")


def test_trim_history_under_budget_keeps_everything() -> None:
    turns = [LlmMessage(role="user", content="short"), LlmMessage(role="assistant", content="ok")]
    assert trim_history(turns, 100) == turns


def test_trim_history_drops_oldest_until_fit() -> None:
    turns = [
        LlmMessage(role="user", content="x" * 400),
        LlmMessage(role="assistant", content="x" * 400),
        LlmMessage(role="user", content="x" * 200),
    ]
    trimmed = trim_history(turns, 200)
    assert trimmed == turns[2:]


def test_trim_history_truncates_single_oversized_turn() -> None:
    turns = [LlmMessage(role="user", content="x" * 500)]
    trimmed = trim_history(turns, 10)
    assert len(trimmed) == 1
    assert trimmed[0].role == "user"
    assert trimmed[0].content == "x" * ((10 - 1) * CHARS_PER_TOKEN)


def test_trim_history_cap_holds_even_when_head_turn_is_oversized() -> None:
    turns = [
        LlmMessage(role="user", content="y" * 4000),
        LlmMessage(role="assistant", content="Summary A."),
    ]
    trimmed = trim_history(turns, 100)
    assert sum(estimate_text_tokens(turn.content) for turn in trimmed) <= 100
    assert trimmed[0].content == "y" * ((100 - 1) * CHARS_PER_TOKEN)
    assert trimmed[1].content == "Sum"


def test_trim_history_never_starts_with_assistant_turn() -> None:
    turns = [
        LlmMessage(role="user", content="a" * 100),
        LlmMessage(role="user", content="b" * 100),
        LlmMessage(role="assistant", content="c" * 100),
    ]
    trimmed = trim_history(turns, 60)
    assert trimmed[0].role == "user"
    assert trimmed[0].content == "b" * 100
    assert trimmed[1].role == "assistant"


def test_trim_history_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        trim_history([], 0)


def test_trim_history_empty_returns_empty() -> None:
    assert trim_history([], 100) == []


def test_session_seed_replaces_and_trims_history() -> None:
    session = ChatSession()
    session.history = [LlmMessage(role="user", content="stale")]
    session.seed(
        [
            message(2, MessageRole.ASSISTANT, "Load stable."),
            message(1, MessageRole.USER, "plan my week"),
        ],
        max_tokens=1000,
    )
    roles = [turn.role for turn in session.history]
    assert roles == ["user", "assistant"]


def test_session_seed_respects_token_cap() -> None:
    session = ChatSession()
    session.seed([message(1, MessageRole.USER, "y" * 4000)], max_tokens=100)
    assert [turn.role for turn in session.history] == ["user"]
    assert session.history[0].content == "y" * ((100 - 1) * CHARS_PER_TOKEN)


def test_session_append_extends_history_in_order() -> None:
    session = ChatSession()
    session.append("question", "reply")
    session.append("follow up", "reply 2")
    assert session.history == [
        LlmMessage(role="user", content="question"),
        LlmMessage(role="assistant", content="reply"),
        LlmMessage(role="user", content="follow up"),
        LlmMessage(role="assistant", content="reply 2"),
    ]


def test_session_append_trims_to_cap() -> None:
    session = ChatSession(cap=100)
    session.append("x" * 40, "y" * 4000)
    assert [turn.role for turn in session.history] == ["user", "assistant"]
    assert session.history[0].content == "x" * 40
    expected = 100 - estimate_text_tokens("x" * 40)
    assert session.history[1].content == "y" * (expected * CHARS_PER_TOKEN)


def test_session_append_without_cap_keeps_everything() -> None:
    session = ChatSession()
    session.append("x" * 400, "y" * 4000)
    assert len(session.history) == 2


def test_trim_history_clamps_at_minimum_cap() -> None:
    turns = [
        LlmMessage(role="user", content="hello"),
        LlmMessage(role="assistant", content="hi"),
    ]
    trimmed = trim_history(turns, 1)
    assert all(turn.content for turn in trimmed)

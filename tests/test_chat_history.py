from datetime import UTC, datetime

import pytest

from open_endurance_coach.chat.history import ChatSession, seed_turns, trim_history
from open_endurance_coach.clients.llm import LlmMessage
from open_endurance_coach.schemas.decisions import DecisionReport
from open_endurance_coach.store.records import Feedback, FeedbackWithReport

NOW = datetime(2024, 2, 1, 12, 0, 0, tzinfo=UTC)


def entry(
    feedback_id: int, draft_id: int, content: str, report: DecisionReport
) -> FeedbackWithReport:
    return FeedbackWithReport(
        feedback=Feedback(id=feedback_id, draft_id=draft_id, created_at=NOW, content=content),
        report=report,
    )


REPORT_A = DecisionReport(summary="Summary A.", findings=["Finding A1.", "Finding A2."])
REPORT_B = DecisionReport(summary="Summary B.", findings=["Finding B1."])


def test_seed_turns_empty() -> None:
    assert seed_turns([]) == []


def test_seed_turns_single_entry_pairs_user_with_assistant() -> None:
    turns = seed_turns([entry(1, 10, "RPE was 7", REPORT_A)])
    assert turns == [
        LlmMessage(role="user", content="RPE was 7"),
        LlmMessage(role="assistant", content="Summary A.\n- Finding A1.\n- Finding A2."),
    ]


def test_seed_turns_multi_round_draft_emits_one_final_reply() -> None:
    turns = seed_turns(
        [
            entry(3, 10, "third answer", REPORT_A),
            entry(2, 10, "second answer", REPORT_A),
            entry(1, 10, "first answer", REPORT_A),
        ]
    )
    assert [turn.role for turn in turns] == ["user", "user", "user", "assistant"]
    assert [turn.content for turn in turns[:3]] == [
        "first answer",
        "second answer",
        "third answer",
    ]
    assert turns[3].content == "Summary A.\n- Finding A1.\n- Finding A2."


def test_seed_turns_groups_consecutive_drafts() -> None:
    turns = seed_turns(
        [
            entry(2, 20, "answer B", REPORT_B),
            entry(1, 10, "answer A", REPORT_A),
        ]
    )
    assert [turn.role for turn in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0].content == "answer A"
    assert turns[1].content.startswith("Summary A.")
    assert turns[2].content == "answer B"
    assert turns[3].content.startswith("Summary B.")


def test_seed_turns_interleaved_drafts_reply_per_group() -> None:
    turns = seed_turns(
        [
            entry(3, 10, "answer A2", REPORT_A),
            entry(2, 20, "answer B", REPORT_B),
            entry(1, 10, "answer A1", REPORT_A),
        ]
    )
    assert [turn.role for turn in turns] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert turns[1].content.startswith("Summary A.")
    assert turns[3].content.startswith("Summary B.")
    assert turns[5].content.startswith("Summary A.")


def test_seed_turns_report_without_findings_is_summary_only() -> None:
    turns = seed_turns([entry(1, 10, "answer", DecisionReport(summary="Bare."))])
    assert turns[1].content == "Bare."


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
    assert trimmed[0].content == "x" * 36


def test_trim_history_cap_holds_even_when_head_turn_is_oversized() -> None:
    turns = [
        LlmMessage(role="user", content="y" * 4000),
        LlmMessage(role="assistant", content="Summary A."),
    ]
    trimmed = trim_history(turns, 100)
    assert sum(max(1, len(turn.content) // 4) for turn in trimmed) <= 100
    assert trimmed[0].content == "y" * 396
    assert trimmed[1].content == "Summ"


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
        [entry(1, 10, "RPE was 7", REPORT_A), entry(2, 10, "also slept badly", REPORT_A)],
        max_tokens=1000,
    )
    roles = [turn.role for turn in session.history]
    assert roles == ["user", "user", "assistant"]


def test_session_seed_respects_token_cap() -> None:
    session = ChatSession()
    session.seed([entry(1, 10, "y" * 4000, REPORT_A)], max_tokens=100)
    assert [turn.role for turn in session.history] == ["user", "assistant"]
    assert session.history[0].content == "y" * 396
    assert session.history[1].content == "Summ"


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
    assert session.history[1].content == "y" * 360


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

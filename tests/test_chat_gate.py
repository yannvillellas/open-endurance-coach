import pytest

from open_endurance_coach.chat.gate import (
    Cancelled,
    Declined,
    Discuss,
    Feedback,
    Ignored,
    PlanSnapshot,
    Proceed,
    handle,
)

APPROVE = PlanSnapshot(
    action="approve", plan_text="Draft #3 - approve these mutations: ...", draft_id=3
)
APPLY = PlanSnapshot(action="apply", plan_text="Decision #1 - write: ...", draft_id=3)
REJECT = PlanSnapshot(action="reject", plan_text="Draft #3 - reject: ...", draft_id=3)


@pytest.mark.parametrize("line", ["yes", "YES", " Yes ", "\tyes\n"])
def test_literal_yes_proceeds(line: str) -> None:
    assert handle(line, APPROVE) == Proceed()


@pytest.mark.parametrize("line", ["no", "NO", " no "])
def test_literal_no_declines(line: str) -> None:
    assert handle(line, APPROVE) == Declined()


@pytest.mark.parametrize("line", ["cancel", "CANCEL", " cancel "])
def test_literal_cancel_exits_confirmation(line: str) -> None:
    assert handle(line, APPROVE) == Cancelled()


@pytest.mark.parametrize("line", ["", "   ", "\t "])
def test_blank_lines_are_ignored(line: str) -> None:
    assert handle(line, APPROVE) == Ignored()


@pytest.mark.parametrize("line", ["y", "n", "yes, but wait", "yes please", "yes.", "no thanks"])
def test_fuzzy_yes_no_never_resolve_the_gate(line: str) -> None:
    assert handle(line, APPROVE) == Feedback(line.strip())


def test_any_other_input_on_approve_falls_back_to_feedback() -> None:
    assert handle("Not yet - explain", APPROVE) == Feedback("Not yet - explain")


def test_any_other_input_on_reject_falls_back_to_feedback() -> None:
    assert handle("Hold on", REJECT) == Feedback("Hold on")


def test_any_other_input_on_apply_falls_back_to_discussion() -> None:
    assert handle("Wait, what does update mean?", APPLY) == Discuss("Wait, what does update mean?")


def test_missing_draft_id_falls_back_to_discussion() -> None:
    snapshot = PlanSnapshot(action="approve", plan_text="plan", draft_id=None)
    assert handle("explain first", snapshot) == Discuss("explain first")


def test_fallback_preserves_interior_whitespace_and_strips_ends() -> None:
    assert handle("  RPE  was  7  ", APPROVE) == Feedback("RPE  was  7")


def test_cancel_with_extra_text_is_a_fallback_not_a_cancel() -> None:
    assert handle("cancel it", APPROVE) == Feedback("cancel it")


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/exit", True),
        ("/quit", True),
        ("/EXIT", True),
        ("  /quit now  ", True),
        ("exit", False),
        ("", False),
        ("   ", False),
        ("yes", False),
    ],
)
def test_is_exit_command(line: str, expected: bool) -> None:
    from open_endurance_coach.chat.gate import is_exit_command

    assert is_exit_command(line) is expected


def test_recoverable_exceptions_include_expected_types() -> None:
    import sqlite3

    from open_endurance_coach.chat.gate import RECOVERABLE_EXCEPTIONS
    from open_endurance_coach.clients.llm import LlmError

    assert LlmError in RECOVERABLE_EXCEPTIONS
    assert ValueError in RECOVERABLE_EXCEPTIONS
    assert RuntimeError in RECOVERABLE_EXCEPTIONS
    assert sqlite3.Error in RECOVERABLE_EXCEPTIONS


def test_exit_aliases_shared_between_dispatch_and_gate() -> None:
    from open_endurance_coach.chat.dispatch import Exit, dispatch
    from open_endurance_coach.chat.gate import EXIT_NAMES, is_exit_command
    from open_endurance_coach.chat.state import ChatState

    assert {"exit", "quit"} == EXIT_NAMES
    for name in EXIT_NAMES:
        assert is_exit_command(f"/{name}") is True
        assert isinstance(dispatch(f"/{name}", ChatState()), Exit)
    assert is_exit_command("/bogus") is False

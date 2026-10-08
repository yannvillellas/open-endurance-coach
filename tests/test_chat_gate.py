from datetime import date

import pytest

from open_endurance_coach.chat.gate import (
    Declined,
    Feedback,
    Ignored,
    InvalidSelection,
    PlanItem,
    PlanSnapshot,
    Proceed,
    ProceedSubset,
    handle,
)

APPROVE = PlanSnapshot(plan_text="Draft #3 - approve these mutations: ...", draft_id=3)

# Displayed as 1: Tue (mutation 2), 2: Thu (mutation 0), 3: Thu (mutation 3), 4: undated (1).
ITEMS = PlanSnapshot(
    plan_text="Apply this to Intervals.icu: ...",
    draft_id=3,
    items=(
        PlanItem(index=2, day=date(2026, 10, 6)),
        PlanItem(index=0, day=date(2026, 10, 8)),
        PlanItem(index=3, day=date(2026, 10, 8)),
        PlanItem(index=1, day=None),
    ),
)


@pytest.mark.parametrize("line", ["yes", "YES", " Yes ", "\tyes\n"])
def test_literal_yes_proceeds(line: str) -> None:
    assert handle(line, APPROVE) == Proceed()


@pytest.mark.parametrize("line", ["no", "NO", " no "])
def test_literal_no_declines(line: str) -> None:
    assert handle(line, APPROVE) == Declined()


@pytest.mark.parametrize("line", ["cancel", "CANCEL", " cancel "])
def test_literal_cancel_is_a_revision_request(line: str) -> None:
    assert handle(line, APPROVE) == Feedback(line.strip())


@pytest.mark.parametrize("line", ["", "   ", "\t "])
def test_blank_lines_are_ignored(line: str) -> None:
    assert handle(line, APPROVE) == Ignored()


@pytest.mark.parametrize("line", ["y", "n", "yes, but wait", "yes please", "yes.", "no thanks"])
def test_fuzzy_yes_no_never_resolve_the_gate(line: str) -> None:
    assert handle(line, APPROVE) == Feedback(line.strip())


@pytest.mark.parametrize("line", ["yes please", "yes, but wait", "YES SIR", "yes except", "yes."])
def test_fuzzy_yes_with_items_is_still_feedback(line: str) -> None:
    assert handle(line, ITEMS) == Feedback(line.strip())


@pytest.mark.parametrize(
    ("line", "indices"),
    [
        ("yes 1", (2,)),
        ("yes 2 4", (0, 1)),
        ("YES 4, 1", (1, 2)),
        ("yes 1 and 3", (2, 3)),
        ("yes thursday", (0, 3)),
        ("yes tue", (2,)),
        ("yes 2026-10-06", (2,)),
        ("yes except 1", (0, 1, 3)),
        ("yes except Thursday", (1, 2)),
        ("Yes except thu, 4", (2,)),
        ("yes except 2026-10-08", (1, 2)),
        ("yes 1,,3", (2, 3)),
        ("yes 1,", (2,)),
        ("yes ,1", (2,)),
        ("yes 1 1", (2,)),
        ("yes except 1,", (0, 1, 3)),
    ],
)
def test_yes_with_a_selection_approves_a_subset(line: str, indices: tuple[int, ...]) -> None:
    assert handle(line, ITEMS) == ProceedSubset(indices)


@pytest.mark.parametrize("line", ["yes 1 2 3 4", "yes 1 thursday 4"])
def test_a_selection_covering_every_item_is_a_full_yes(line: str) -> None:
    assert handle(line, ITEMS) == Proceed()


@pytest.mark.parametrize(
    "line",
    ["yes 5", "yes 0", "yes friday", "yes 2026-10-09", "yes except friday", "yes 1 5"],
)
def test_a_selection_matching_nothing_never_proceeds(line: str) -> None:
    assert isinstance(handle(line, ITEMS), InvalidSelection)


@pytest.mark.parametrize(
    "line",
    ["yes ²", "yes ①", "yes ٣", "yes 1 ²", "yes \uff11", "yes 1\uff12"],
)
def test_a_non_ascii_digit_never_proceeds(line: str) -> None:
    assert isinstance(handle(line, ITEMS), InvalidSelection)


def test_an_over_long_numeric_token_never_proceeds() -> None:
    assert isinstance(handle("yes " + "9" * 4301, ITEMS), InvalidSelection)


def test_excluding_every_item_never_proceeds() -> None:
    result = handle("yes except 1 thursday 4", ITEMS)
    assert isinstance(result, InvalidSelection)


def test_a_selection_without_items_never_proceeds() -> None:
    assert isinstance(handle("yes except 1", APPROVE), InvalidSelection)


@pytest.mark.parametrize("line", ["yes ,", "yes ,,", "yes except ,", "yes except ,,"])
def test_a_selection_with_only_empty_tokens_never_proceeds(line: str) -> None:
    assert isinstance(handle(line, ITEMS), InvalidSelection)


@pytest.mark.parametrize("line", ["yes 1 bogus", "yes bogus 1"])
def test_mixed_valid_and_invalid_tokens_fall_back_to_feedback(line: str) -> None:
    assert handle(line, ITEMS) == Feedback(line.strip())


def test_a_selector_without_items_never_proceeds() -> None:
    assert isinstance(handle("yes 1", APPROVE), InvalidSelection)


def test_plain_yes_with_items_still_approves_everything() -> None:
    assert handle("yes", ITEMS) == Proceed()


def test_any_other_input_on_approve_falls_back_to_feedback() -> None:
    assert handle("Not yet - explain", APPROVE) == Feedback("Not yet - explain")


def test_any_other_input_on_reject_falls_back_to_feedback() -> None:
    assert handle("Hold on", APPROVE) == Feedback("Hold on")


def test_fallback_preserves_interior_whitespace_and_strips_ends() -> None:
    assert handle("  RPE  was  7  ", APPROVE) == Feedback("RPE  was  7")


def test_cancel_with_extra_text_is_a_revision_request() -> None:
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
    from open_endurance_coach.clients.intervals import IntervalsApiError
    from open_endurance_coach.clients.llm import LlmError
    from open_endurance_coach.writer.calendar import WriterError

    assert LlmError in RECOVERABLE_EXCEPTIONS
    assert IntervalsApiError in RECOVERABLE_EXCEPTIONS
    assert WriterError in RECOVERABLE_EXCEPTIONS
    assert ValueError in RECOVERABLE_EXCEPTIONS
    assert sqlite3.Error in RECOVERABLE_EXCEPTIONS
    assert RuntimeError not in RECOVERABLE_EXCEPTIONS


def test_exit_aliases_shared_between_dispatch_and_gate() -> None:
    from open_endurance_coach.chat.dispatch import Exit, dispatch
    from open_endurance_coach.chat.gate import EXIT_NAMES, is_exit_command
    from open_endurance_coach.chat.state import ChatState

    assert {"exit", "quit"} == EXIT_NAMES
    for name in EXIT_NAMES:
        assert is_exit_command(f"/{name}") is True
        assert isinstance(dispatch(f"/{name}", ChatState()), Exit)
    assert is_exit_command("/bogus") is False

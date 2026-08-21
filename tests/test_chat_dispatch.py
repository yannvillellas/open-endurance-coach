import pytest

from open_endurance_coach.chat.dispatch import (
    Command,
    Confirmation,
    Converse,
    Exit,
    Ignore,
    UnknownCommand,
    dispatch,
)
from open_endurance_coach.chat.gate import PlanSnapshot
from open_endurance_coach.chat.state import ChatState

_CONFIRMING = ChatState(
    plan=PlanSnapshot(action="approve", plan_text="Draft #3 - approve", draft_id=3)
)


@pytest.mark.parametrize(
    ("line", "name", "args"),
    [
        ("/help", "help", []),
        ("/analyze", "analyze", []),
        ("/analyze how was my week", "analyze", ["how", "was", "my", "week"]),
        ("/review", "review", []),
        ("/review 3", "review", ["3"]),
        ("/feedback 1 RPE 7", "feedback", ["1", "RPE", "7"]),
        ("/approve 3", "approve", ["3"]),
        ("/reject 2", "reject", ["2"]),
        ("/apply", "apply", []),
        ("/apply 5 --write", "apply", ["5", "--write"]),
    ],
)
def test_slash_commands_dispatch_to_commands(line: str, name: str, args: list[str]) -> None:
    assert dispatch(line, ChatState()) == Command(name, args)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("/APPROVE 3", Command("approve", ["3"])),
        ("/Approve 3", Command("approve", ["3"])),
        ("/REVIEW", Command("review", [])),
    ],
)
def test_command_names_are_case_insensitive(line: str, expected: Command) -> None:
    assert dispatch(line, ChatState()) == expected


@pytest.mark.parametrize("line", ["/exit", "/quit", "/EXIT", "  /quit  "])
def test_exit_commands(line: str) -> None:
    assert dispatch(line, ChatState()) == Exit()


@pytest.mark.parametrize("line", ["/bogus", "/feedbackx 1 hi", "/", "  /  "])
def test_unknown_slash_commands_are_reported_verbatim(line: str) -> None:
    assert dispatch(line, ChatState()) == UnknownCommand(line.strip())


@pytest.mark.parametrize(
    ("line", "text"),
    [
        ("hi", "hi"),
        ("  how did my HR look?  ", "how did my HR look?"),
        ("yes", "yes"),
        ("RPE was 7 yesterday", "RPE was 7 yesterday"),
    ],
)
def test_free_text_routes_to_converse(line: str, text: str) -> None:
    assert dispatch(line, ChatState()) == Converse(text)


@pytest.mark.parametrize("line", ["", "   ", "\t "])
def test_blank_lines_are_ignored(line: str) -> None:
    assert dispatch(line, ChatState()) == Ignore()


@pytest.mark.parametrize(
    "line",
    ["yes", "no", "cancel", "YES", "yes, but wait", "any free text", "/exit", "/approve 3"],
)
def test_confirming_state_consumes_every_line_before_routing(line: str) -> None:
    assert dispatch(line, _CONFIRMING) == Confirmation(line.strip())


def test_blank_line_ignored_even_in_confirming_state() -> None:
    assert dispatch("   ", _CONFIRMING) == Ignore()

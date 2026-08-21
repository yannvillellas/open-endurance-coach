from dataclasses import dataclass, field

from open_endurance_coach.clients.llm import LlmMessage
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport
from open_endurance_coach.store.records import FeedbackWithReport


def _tokens_of(turn: LlmMessage) -> int:
    return max(1, len(turn.content) // 4)


def _assistant_turn(report: DecisionReport) -> LlmMessage:
    content = report.summary
    if report.findings:
        content += "\n" + "\n".join(f"- {finding}" for finding in report.findings)
    return LlmMessage(role="assistant", content=content)


def seed_turns(entries: list[FeedbackWithReport]) -> list[LlmMessage]:
    turns: list[LlmMessage] = []
    count = len(entries)
    for position, entry in enumerate(reversed(entries)):
        turns.append(LlmMessage(role="user", content=entry.feedback.content))
        next_draft = (
            entries[count - 2 - position].feedback.draft_id if position + 1 < count else None
        )
        if next_draft != entry.feedback.draft_id:
            turns.append(_assistant_turn(entry.report))
    return turns


def trim_history(turns: list[LlmMessage], max_tokens: int) -> list[LlmMessage]:
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be positive: {max_tokens}")
    remaining = list(turns)
    while len(remaining) > 1 and sum(_tokens_of(turn) for turn in remaining) > max_tokens:
        remaining.pop(0)
    if remaining and _tokens_of(remaining[0]) > max_tokens:
        remaining[0] = LlmMessage(
            role=remaining[0].role, content=remaining[0].content[: max_tokens * 4]
        )
    return remaining


@dataclass
class ChatSession:
    history: list[LlmMessage] = field(default_factory=list)
    context: CoachContext | None = None

    def seed(self, entries: list[FeedbackWithReport], *, max_tokens: int) -> None:
        self.history = trim_history(seed_turns(entries), max_tokens)

    def append(self, user_text: str, assistant_text: str) -> None:
        self.history.append(LlmMessage(role="user", content=user_text))
        self.history.append(LlmMessage(role="assistant", content=assistant_text))

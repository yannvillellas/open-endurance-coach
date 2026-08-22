from dataclasses import dataclass, field

from open_endurance_coach.clients.llm import LlmMessage
from open_endurance_coach.schemas.context import CoachContext
from open_endurance_coach.schemas.decisions import DecisionReport
from open_endurance_coach.store.records import FeedbackWithReport


def _tokens_of(turn: LlmMessage) -> int:
    return max(1, len(turn.content) // 4)


def assistant_turn(report: DecisionReport) -> LlmMessage:
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
            turns.append(assistant_turn(entry.report))
    return turns


def trim_history(turns: list[LlmMessage], max_tokens: int) -> list[LlmMessage]:
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be positive: {max_tokens}")
    remaining = list(turns)
    pairs = [remaining[i : i + 2] for i in range(0, len(remaining), 2)]
    while len(pairs) > 1 and sum(_tokens_of(turn) for pair in pairs for turn in pair) > max_tokens:
        pairs.pop(0)
    kept = [turn for pair in pairs for turn in pair]
    if kept and sum(_tokens_of(turn) for turn in kept) > max_tokens:
        head = sum(_tokens_of(turn) for turn in kept[:-1])
        kept[-1] = LlmMessage(
            role=kept[-1].role,
            content=kept[-1].content[: max(1, (max_tokens - head) * 4)],
        )
    return kept


@dataclass
class ChatSession:
    history: list[LlmMessage] = field(default_factory=list)
    context: CoachContext | None = None
    cap: int | None = None

    def seed(self, entries: list[FeedbackWithReport], *, max_tokens: int) -> None:
        self.history = trim_history(seed_turns(entries), max_tokens)

    def append(self, user_text: str, assistant_text: str) -> None:
        self.history.append(LlmMessage(role="user", content=user_text))
        self.history.append(LlmMessage(role="assistant", content=assistant_text))
        if self.cap is not None:
            self.history = trim_history(self.history, self.cap)

from dataclasses import dataclass
from enum import StrEnum

from open_endurance_coach.chat.gate import PlanSnapshot


class ChatMode(StrEnum):
    CONVERSING = "conversing"
    CONFIRMING = "confirming"


@dataclass(frozen=True)
class ChatState:
    plan: PlanSnapshot | None = None

    @property
    def mode(self) -> ChatMode:
        return ChatMode.CONFIRMING if self.plan is not None else ChatMode.CONVERSING

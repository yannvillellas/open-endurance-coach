from dataclasses import dataclass
from enum import StrEnum


class ChatMode(StrEnum):
    CONVERSING = "conversing"
    CONFIRMING = "confirming"


@dataclass(frozen=True)
class ChatState:
    mode: ChatMode = ChatMode.CONVERSING

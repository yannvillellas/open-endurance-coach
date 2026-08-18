from datetime import datetime
from typing import Any

from open_endurance_coach.clients.llm import LlmCompletion


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class FakeLlmProvider:
    name = "fake"

    def __init__(self, responses: list[LlmCompletion] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        model: str,
        messages: list,
        thinking: bool,
        json_mode: bool,
        max_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None,
    ) -> LlmCompletion:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "thinking": thinking,
                "json_mode": json_mode,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            }
        )
        if not self.responses:
            raise AssertionError("FakeLlmProvider: no responses left")
        return self.responses.pop(0)


def completion(content: str, reasoning_content: str | None = None) -> LlmCompletion:
    return LlmCompletion(content=content, reasoning_content=reasoning_content, model="fake-model")

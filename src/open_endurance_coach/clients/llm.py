import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from open_endurance_coach.config import Settings


class LlmError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlmMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LlmCompletion:
    content: str
    reasoning_content: str | None = None
    model: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)


class LlmProvider(Protocol):
    name: str

    async def complete(
        self,
        *,
        model: str,
        messages: list[LlmMessage],
        thinking: bool,
        json_mode: bool,
        max_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None,
    ) -> LlmCompletion: ...


class LlmClient:
    def __init__(
        self,
        settings: Settings,
        providers: Mapping[str, LlmProvider],
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._settings = settings
        self._providers = providers
        self._sleep = sleep or asyncio.sleep
        if settings.llm_provider not in providers:
            raise LlmError(
                f"Unknown LLM provider: {settings.llm_provider!r} "
                f"(available: {', '.join(sorted(providers))})"
            )

    async def complete(
        self,
        messages: list[LlmMessage],
        *,
        json_mode: bool = False,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> LlmCompletion:
        provider = self._providers[self._settings.llm_provider]
        return await provider.complete(
            model=self._settings.llm_model,
            messages=messages,
            thinking=self._settings.llm_thinking if thinking is None else thinking,
            json_mode=json_mode,
            max_tokens=max_tokens if max_tokens is not None else self._settings.llm_max_tokens,
            temperature=self._settings.llm_temperature if temperature is None else temperature,
            reasoning_effort=(
                self._settings.llm_reasoning_effort
                if reasoning_effort is None
                else reasoning_effort
            ),
        )

    async def complete_json(
        self,
        messages: list[LlmMessage],
        *,
        max_attempts: int | None = None,
    ) -> str:
        attempts = self._settings.max_retries if max_attempts is None else max_attempts
        last_error: LlmError | None = None
        for attempt in range(attempts):
            completion = await self.complete(messages, json_mode=True)
            content = completion.content
            if not content or not content.strip():
                last_error = LlmError("empty content returned")
            else:
                try:
                    json.loads(content)
                except json.JSONDecodeError as exc:
                    last_error = LlmError(f"invalid JSON returned: {exc}")
                else:
                    return content
            await self._sleep(2**attempt)
        raise LlmError(f"JSON completion failed after {attempts} attempts: {last_error}")

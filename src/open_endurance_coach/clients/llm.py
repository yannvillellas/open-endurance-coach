import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from open_endurance_coach.config import Settings, describe_providers
from open_endurance_coach.tokens import estimate_text_tokens

logger = logging.getLogger(__name__)

TOKEN_ESTIMATE_DRIFT_THRESHOLD = 0.25


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
    finish_reason: str | None = None


def _completion_diagnostics(completion: LlmCompletion) -> str:
    usage = completion.usage or {}
    return (
        f"finish_reason={completion.finish_reason or 'unknown'},"
        f" completion_tokens={usage.get('completion_tokens', 'unknown')},"
        f" reasoning_content={'present' if completion.reasoning_content else 'absent'}"
    )


def _output_budget_error(completion: LlmCompletion) -> str:
    return (
        "the model exhausted its output budget before producing an answer"
        f" ({_completion_diagnostics(completion)});"
        " raise LLM_MAX_TOKENS or lower LLM_REASONING_EFFORT"
    )


def _empty_content_error(completion: LlmCompletion) -> str:
    return f"empty content returned ({_completion_diagnostics(completion)})"


def _warn_on_token_estimate_drift(messages: list[LlmMessage], completion: LlmCompletion) -> None:
    prompt_tokens = (completion.usage or {}).get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return
    estimated = sum(estimate_text_tokens(message.content) for message in messages)
    drift = abs(estimated - prompt_tokens) / prompt_tokens
    if drift > TOKEN_ESTIMATE_DRIFT_THRESHOLD:
        logger.warning(
            "token estimate drift: estimated=%d actual=%d (%.0f%%) model=%s",
            estimated,
            prompt_tokens,
            drift * 100,
            completion.model,
        )


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

    async def aclose(self) -> None: ...


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
            available = describe_providers(providers, settings)
            raise LlmError(
                f"Unknown LLM provider: {settings.llm_provider!r}\n"
                f"Available providers:\n{available}"
            )

    @property
    def provider_name(self) -> str:
        return self._settings.llm_provider

    @property
    def model_name(self) -> str:
        return self._settings.llm_model

    def select(self, *, provider: str | None = None, model: str | None = None) -> None:
        if provider is not None and provider not in self._providers:
            available = describe_providers(self._providers, self._settings)
            raise LlmError(f"Unknown LLM provider: {provider!r}\nAvailable providers:\n{available}")
        self._settings = self._settings.with_llm_override(provider=provider, model=model)

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
        completion = await provider.complete(
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
        cached = (completion.usage or {}).get("prompt_cache_hit_tokens")
        if cached is not None:
            logger.debug("prompt cache hit tokens: %s", cached)
        _warn_on_token_estimate_drift(messages, completion)
        return completion

    async def complete_json(
        self,
        messages: list[LlmMessage],
        *,
        max_attempts: int | None = None,
        validator: Callable[[Any], Any] | None = None,
    ) -> str:
        attempts = self._settings.max_retries if max_attempts is None else max_attempts
        attempts = max(1, attempts)
        last_error: LlmError | None = None
        for attempt in range(attempts):
            completion = await self.complete(messages, json_mode=True)
            content = completion.content
            if content and content.strip():
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError as exc:
                    last_error = LlmError(f"invalid JSON returned: {exc}")
                else:
                    if validator is not None:
                        try:
                            validator(payload)
                        except Exception as exc:
                            last_error = LlmError(f"JSON validation failed: {exc}")
                        else:
                            return content
                    else:
                        return content
            else:
                last_error = LlmError(_empty_content_error(completion))
            if completion.finish_reason == "length":
                raise LlmError(_output_budget_error(completion))
            await self._sleep(2**attempt)
        raise LlmError(f"JSON completion failed after {attempts} attempts: {last_error}")

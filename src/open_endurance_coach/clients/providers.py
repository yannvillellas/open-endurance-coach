import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx

from open_endurance_coach.config import Settings

from .http import parse_retry_after
from .llm import LlmCompletion, LlmError, LlmMessage, LlmProvider

OVH_MIN_429_BACKOFF_SECONDS = 60.0


class _OpenAiCompatibleProvider:
    name = ""
    _error_label = "LLM"
    _min_429_backoff = 0.0
    _rate_limit_hint: str | None = None
    _sends_thinking = True

    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str,
        headers: Mapping[str, str],
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._settings = settings
        self._sleep = sleep or asyncio.sleep
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=dict(headers),
            timeout=settings.llm_timeout_seconds,
            transport=transport,
        )

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
    ) -> LlmCompletion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
        }
        if self._sends_thinking:
            payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
            if thinking and reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
            if not thinking and temperature is not None:
                payload["temperature"] = temperature
        elif temperature is not None:
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response: httpx.Response | None = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.TransportError:
                response = None
                if attempt < self._settings.max_retries:
                    await self._sleep(self._settings.retry_base_delay * 2**attempt)
                continue
            if response.status_code == 429 and attempt < self._settings.max_retries:
                backoff = self._settings.retry_base_delay * 2**attempt
                if response.headers.get("Retry-After") is not None:
                    wait = parse_retry_after(response.headers, default=self._min_429_backoff)
                else:
                    wait = max(backoff, self._min_429_backoff)
                await self._sleep(wait)
                continue
            if response.status_code >= 500 and attempt < self._settings.max_retries:
                await self._sleep(self._settings.retry_base_delay * 2**attempt)
                continue
            if response.status_code >= 400:
                message = (
                    f"{self._error_label} API error {response.status_code}: {response.text[:500]}"
                )
                if response.status_code == 429 and self._rate_limit_hint:
                    message = f"{message}\n{self._rate_limit_hint}"
                raise LlmError(message)
            break
        if response is None:
            raise LlmError(f"{self._error_label} API unreachable after retries")
        try:
            data = response.json()
        except ValueError as exc:
            raise LlmError(
                f"{self._error_label} returned non-JSON response: {response.text[:200]}"
            ) from exc
        try:
            choice = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(
                f"unexpected {self._error_label} response shape: {str(data)[:200]}"
            ) from exc
        if not isinstance(choice, dict):
            raise LlmError(f"unexpected {self._error_label} response shape: {str(data)[:200]}")
        return LlmCompletion(
            content=choice.get("content") or "",
            reasoning_content=choice.get("reasoning_content"),
            model=data.get("model", model),
            usage=data.get("usage") or {},
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class DeepSeekProvider(_OpenAiCompatibleProvider):
    name = "deepseek"
    _error_label = "DeepSeek"

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(
            settings,
            base_url=settings.deepseek_base_url,
            headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
            transport=transport,
            sleep=sleep,
        )


class OvhProvider(_OpenAiCompatibleProvider):
    name = "ovh"
    _error_label = "OVHcloud AI Endpoints"
    _min_429_backoff = OVH_MIN_429_BACKOFF_SECONDS
    _sends_thinking = False
    _rate_limit_hint = (
        "The OVHcloud anonymous free tier allows about 2 requests/minute per IP."
        " Wait a minute, set OVH_API_KEY for the paid tier, or select another provider"
        " with --provider <name>."
    )

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        headers = (
            {"Authorization": f"Bearer {settings.ovh_api_key}"} if settings.ovh_api_key else {}
        )
        super().__init__(
            settings,
            base_url=settings.ovh_base_url,
            headers=headers,
            transport=transport,
            sleep=sleep,
        )


def build_registry(
    settings: Settings,
    transports: Mapping[str, httpx.AsyncBaseTransport] | None = None,
) -> dict[str, LlmProvider]:
    transports = transports or {}
    return {
        "ovh": OvhProvider(settings, transports.get("ovh")),
        "deepseek": DeepSeekProvider(settings, transports.get("deepseek")),
    }

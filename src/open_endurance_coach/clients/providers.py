from collections.abc import Mapping
from typing import Any

import httpx

from open_endurance_coach.config import Settings

from .llm import LlmCompletion, LlmError, LlmMessage, LlmProvider


class DeepSeekProvider:
    name = "deepseek"

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.deepseek_base_url,
            headers={"Authorization": f"Bearer {settings.deepseek_api_key}"},
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
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        if thinking and reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if not thinking and temperature is not None:
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            raise LlmError(f"DeepSeek API error {response.status_code}: {response.text[:500]}")
        data = response.json()
        choice = data["choices"][0]["message"]
        return LlmCompletion(
            content=choice.get("content") or "",
            reasoning_content=choice.get("reasoning_content"),
            model=data.get("model", model),
            usage=data.get("usage") or {},
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def build_registry(
    settings: Settings,
    transports: Mapping[str, httpx.AsyncBaseTransport] | None = None,
) -> dict[str, LlmProvider]:
    transports = transports or {}
    return {"deepseek": DeepSeekProvider(settings, transports.get("deepseek"))}

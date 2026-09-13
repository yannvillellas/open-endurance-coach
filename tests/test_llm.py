import json
from typing import Any

import httpx
import pytest

from open_endurance_coach.clients.llm import LlmClient, LlmError, LlmMessage
from open_endurance_coach.clients.providers import DeepSeekProvider, OvhProvider
from open_endurance_coach.config import Settings

from .fakes import FakeLlmProvider, RecordingSleep, completion


def make_client(
    settings: Settings,
    provider: FakeLlmProvider,
    sleep: RecordingSleep | None = None,
) -> LlmClient:
    return LlmClient(settings, {"fake": provider}, sleep=sleep or RecordingSleep())


async def test_provider_selected_from_settings(settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake"})
    provider = FakeLlmProvider([completion("ok")])
    client = make_client(settings, provider)
    result = await client.complete([LlmMessage(role="user", content="hi")])
    assert result.content == "ok"
    assert provider.calls[0]["model"] == settings.llm_model


async def test_unknown_provider_rejected(settings: Settings) -> None:
    with pytest.raises(LlmError, match="Unknown LLM provider"):
        LlmClient(settings, {"fake": FakeLlmProvider()})


async def test_thinking_flag_from_settings(settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake"})
    provider = FakeLlmProvider([completion("ok"), completion("ok")])
    client = make_client(settings, provider)
    await client.complete([LlmMessage(role="user", content="hi")])
    assert provider.calls[0]["thinking"] is True

    await client.complete([LlmMessage(role="user", content="hi")], thinking=False)
    assert provider.calls[-1]["thinking"] is False


async def test_json_mode_flag_forwarded(settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake"})
    provider = FakeLlmProvider([completion('{"a": 1}')])
    client = make_client(settings, provider)
    result = await client.complete_json([LlmMessage(role="user", content="parse this json")])
    assert provider.calls[0]["json_mode"] is True
    assert result == '{"a": 1}'


async def test_complete_json_retries_on_empty_content(settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake"})
    sleep = RecordingSleep()
    provider = FakeLlmProvider([completion(""), completion('{"a": 1}')])
    client = make_client(settings, provider, sleep=sleep)
    result = await client.complete_json([LlmMessage(role="user", content="json please")])
    assert len(provider.calls) == 2
    assert sleep.calls == [1.0]
    assert result == '{"a": 1}'


async def test_complete_json_retries_on_invalid_json(settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake"})
    sleep = RecordingSleep()
    provider = FakeLlmProvider([completion("not json"), completion('{"ok": true}')])
    client = make_client(settings, provider, sleep=sleep)
    result = await client.complete_json([LlmMessage(role="user", content="json please")])
    assert len(provider.calls) == 2
    assert sleep.calls == [1.0]
    assert result == '{"ok": true}'


async def test_complete_json_exhausts_attempts(settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake", "max_retries": 3})
    provider = FakeLlmProvider([completion(""), completion(""), completion("")])
    client = make_client(settings, provider)
    with pytest.raises(LlmError, match="failed after 3 attempts"):
        await client.complete_json([LlmMessage(role="user", content="json please")])
    assert len(provider.calls) == 3


async def test_complete_json_max_attempts_zero_still_attempts_once(settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake"})
    provider = FakeLlmProvider([completion('{"a": 1}')])
    client = make_client(settings, provider)
    result = await client.complete_json(
        [LlmMessage(role="user", content="json please")], max_attempts=0
    )
    assert result == '{"a": 1}'
    assert len(provider.calls) == 1


def require_summary(payload: Any) -> None:
    if not isinstance(payload, dict) or not payload.get("summary"):
        raise ValueError("missing summary")


async def test_complete_json_returns_content_when_validator_accepts(settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake"})
    provider = FakeLlmProvider([completion('{"summary": "ok"}')])
    client = make_client(settings, provider)
    result = await client.complete_json(
        [LlmMessage(role="user", content="json please")], validator=require_summary
    )
    assert result == '{"summary": "ok"}'
    assert len(provider.calls) == 1


async def test_complete_json_retries_when_validator_fails(settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake"})
    sleep = RecordingSleep()
    provider = FakeLlmProvider([completion('{"bad": 1}'), completion('{"summary": "ok"}')])
    client = make_client(settings, provider, sleep=sleep)
    result = await client.complete_json(
        [LlmMessage(role="user", content="json please")], validator=require_summary
    )
    assert len(provider.calls) == 2
    assert sleep.calls == [1.0]
    assert result == '{"summary": "ok"}'


async def test_complete_json_exhausts_attempts_when_validator_always_fails(
    settings: Settings,
) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake", "max_retries": 3})
    provider = FakeLlmProvider([completion('{"bad": 1}')] * 3)
    client = make_client(settings, provider)
    with pytest.raises(LlmError, match="failed after 3 attempts"):
        await client.complete_json(
            [LlmMessage(role="user", content="json please")], validator=require_summary
        )
    assert len(provider.calls) == 3


async def test_complete_json_validator_receives_parsed_payload(settings: Settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "fake"})
    seen: list[Any] = []

    def capture(payload: Any) -> None:
        seen.append(payload)

    provider = FakeLlmProvider([completion(json.dumps({"nested": {"a": [1, 2]}}))])
    client = make_client(settings, provider)
    await client.complete_json([LlmMessage(role="user", content="json please")], validator=capture)
    assert seen == [{"nested": {"a": [1, 2]}}]


def make_provider(
    settings: Settings, handler: Any, sleep: RecordingSleep | None = None
) -> tuple[DeepSeekProvider, RecordingSleep]:
    sleep = sleep or RecordingSleep()
    return DeepSeekProvider(settings, transport=httpx.MockTransport(handler), sleep=sleep), sleep


def ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "model": "deepseek-flash",
            "usage": {},
        },
    )


async def test_provider_retries_on_429_then_succeeds(settings: Settings) -> None:
    responses = [httpx.Response(429, json={}), ok_response()]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    provider, sleep = make_provider(settings, handler)
    completion = await provider.complete(
        model="deepseek-flash",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert completion.content == "ok"
    assert responses == []
    assert sleep.calls == [0.0]


async def test_provider_honors_retry_after_on_429(settings: Settings) -> None:
    responses = [
        httpx.Response(429, headers={"Retry-After": "5"}, json={}),
        ok_response(),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    provider, sleep = make_provider(settings, handler)
    completion = await provider.complete(
        model="deepseek-flash",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert completion.content == "ok"
    assert sleep.calls == [5.0]


async def test_provider_429_http_date_retry_after_falls_back(settings: Settings) -> None:
    responses = [
        httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, json={}),
        ok_response(),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    provider, sleep = make_provider(settings, handler)
    completion = await provider.complete(
        model="deepseek-flash",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert completion.content == "ok"
    assert sleep.calls == [0.0]


async def test_provider_retries_on_500_then_succeeds(settings: Settings) -> None:
    responses = [httpx.Response(500, json={}), ok_response()]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    provider, _ = make_provider(settings, handler)
    completion = await provider.complete(
        model="deepseek-flash",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert completion.content == "ok"


async def test_provider_raises_immediately_on_client_error(settings: Settings) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, json={"error": "bad key"})

    provider, _ = make_provider(settings, handler)
    with pytest.raises(LlmError, match="401"):
        await provider.complete(
            model="deepseek-flash",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )
    assert len(calls) == 1


async def test_provider_retries_network_errors(settings: Settings) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection reset")
        return ok_response()

    provider, _ = make_provider(settings, handler)
    completion = await provider.complete(
        model="deepseek-flash",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert completion.content == "ok"
    assert len(calls) == 2


async def test_provider_network_exhaustion_sleeps_only_between_attempts(
    settings: Settings,
) -> None:
    calls: list[httpx.Request] = []
    sleep = RecordingSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("connection reset")

    provider, _ = make_provider(settings, handler, sleep=sleep)
    with pytest.raises(LlmError, match="unreachable after retries"):
        await provider.complete(
            model="deepseek-flash",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )
    assert len(calls) == 4
    assert sleep.calls == [0.0, 0.0, 0.0]


async def test_provider_429_then_network_error_reports_unreachable(settings: Settings) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={})
        raise httpx.ConnectError("connection reset")

    provider, _ = make_provider(settings, handler)
    with pytest.raises(LlmError, match="unreachable after retries"):
        await provider.complete(
            model="deepseek-flash",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )
    assert len(calls) == 4


async def test_provider_raises_after_retries_exhausted(settings: Settings) -> None:
    calls: list[httpx.Request] = []
    sleep = RecordingSleep()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429, json={})

    provider, _ = make_provider(settings, handler, sleep=sleep)
    with pytest.raises(LlmError, match="API error 429"):
        await provider.complete(
            model="deepseek-flash",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )
    assert len(calls) == 4
    assert sleep.calls == [0.0, 0.0, 0.0]


async def test_provider_guards_malformed_success_response(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "boom"})

    provider, _ = make_provider(settings, handler)
    with pytest.raises(LlmError, match="shape"):
        await provider.complete(
            model="deepseek-flash",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )


async def test_provider_guards_non_dict_message(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": "plain string"}]})

    provider, _ = make_provider(settings, handler)
    with pytest.raises(LlmError, match="shape"):
        await provider.complete(
            model="deepseek-flash",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )


async def test_provider_guards_non_json_body(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway error</html>")

    provider, _ = make_provider(settings, handler)
    with pytest.raises(LlmError, match="non-JSON"):
        await provider.complete(
            model="deepseek-flash",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )


def make_ovh_provider(
    settings: Settings, handler: Any, sleep: RecordingSleep | None = None
) -> tuple[OvhProvider, RecordingSleep]:
    sleep = sleep or RecordingSleep()
    return OvhProvider(settings, transport=httpx.MockTransport(handler), sleep=sleep), sleep


def ovh_ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "model": "Qwen3.5-397B-A17B",
            "usage": {},
        },
    )


async def test_ovh_provider_sends_bearer_auth(settings: Settings) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return ovh_ok_response()

    settings = settings.model_copy(
        update={
            "ovh_api_key": "test-key-123",
            "ovh_base_url": "https://test.ovh.net/v1",
            "llm_provider": "ovh",
            "llm_model": "Qwen3.5-397B-A17B",
        }
    )
    provider, _ = make_ovh_provider(settings, handler)
    await provider.complete(
        model="Qwen3.5-397B-A17B",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert len(captured) == 1
    assert captured[0].headers["Authorization"] == "Bearer test-key-123"


async def test_ovh_provider_omits_auth_header_without_key(settings: Settings) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return ovh_ok_response()

    settings = settings.model_copy(
        update={"ovh_api_key": "", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, _ = make_ovh_provider(settings, handler)
    await provider.complete(
        model="Qwen3.5-397B-A17B",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert len(captured) == 1
    assert "Authorization" not in captured[0].headers


async def test_ovh_provider_payload_structure(settings: Settings) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return ovh_ok_response()

    settings = settings.model_copy(
        update={"ovh_api_key": "key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, _ = make_ovh_provider(settings, handler)
    await provider.complete(
        model="Qwen3.5-397B-A17B",
        messages=[LlmMessage(role="user", content="hello")],
        thinking=True,
        json_mode=False,
        max_tokens=200,
        temperature=None,
        reasoning_effort="high",
    )
    assert len(captured) == 1
    payload = json.loads(captured[0].content.decode())
    assert payload["model"] == "Qwen3.5-397B-A17B"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == 200
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload


async def test_ovh_provider_temperature_without_thinking_param(settings: Settings) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return ovh_ok_response()

    settings = settings.model_copy(
        update={"ovh_api_key": "key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, _ = make_ovh_provider(settings, handler)
    await provider.complete(
        model="Qwen3.5-397B-A17B",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=False,
        json_mode=False,
        max_tokens=100,
        temperature=0.7,
        reasoning_effort=None,
    )
    payload = json.loads(captured[0].content.decode())
    assert "thinking" not in payload
    assert payload["temperature"] == 0.7
    assert "reasoning_effort" not in payload


async def test_ovh_provider_json_mode_flag(settings: Settings) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return ovh_ok_response()

    settings = settings.model_copy(
        update={"ovh_api_key": "key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, _ = make_ovh_provider(settings, handler)
    await provider.complete(
        model="Qwen3.5-397B-A17B",
        messages=[LlmMessage(role="user", content="json please")],
        thinking=True,
        json_mode=True,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    payload = json.loads(captured[0].content.decode())
    assert payload["response_format"] == {"type": "json_object"}


async def test_ovh_provider_retries_on_429(settings: Settings) -> None:
    responses = [httpx.Response(429, json={}), ovh_ok_response()]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    settings = settings.model_copy(
        update={"ovh_api_key": "key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, sleep = make_ovh_provider(settings, handler)
    completion = await provider.complete(
        model="Qwen3.5-397B-A17B",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert completion.content == "ok"
    assert sleep.calls == [60.0]


async def test_ovh_provider_429_honors_larger_retry_after(settings: Settings) -> None:
    responses = [httpx.Response(429, headers={"Retry-After": "120"}, json={}), ovh_ok_response()]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    settings = settings.model_copy(
        update={"ovh_api_key": "key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, sleep = make_ovh_provider(settings, handler)
    completion = await provider.complete(
        model="Qwen3.5-397B-A17B",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert completion.content == "ok"
    assert sleep.calls == [120.0]


async def test_ovh_provider_429_honors_small_server_retry_after(settings: Settings) -> None:
    responses = [httpx.Response(429, headers={"Retry-After": "18"}, json={}), ovh_ok_response()]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    settings = settings.model_copy(
        update={"ovh_api_key": "key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, sleep = make_ovh_provider(settings, handler)
    completion = await provider.complete(
        model="Qwen3.5-397B-A17B",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert completion.content == "ok"
    assert sleep.calls == [18.0]


async def test_ovh_provider_exhausted_429_reports_actionable_hint(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "API rate limit exceeded"})

    settings = settings.model_copy(
        update={
            "ovh_api_key": "key",
            "llm_provider": "ovh",
            "llm_model": "Qwen3.5-397B-A17B",
            "max_retries": 0,
        }
    )
    provider, sleep = make_ovh_provider(settings, handler)
    with pytest.raises(LlmError) as excinfo:
        await provider.complete(
            model="Qwen3.5-397B-A17B",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )
    message = str(excinfo.value)
    assert "429" in message
    assert "OVH_API_KEY" in message
    assert "another provider" in message
    assert sleep.calls == []


async def test_ovh_provider_guards_non_json_body(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway error</html>")

    settings = settings.model_copy(
        update={"ovh_api_key": "key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, _ = make_ovh_provider(settings, handler)
    with pytest.raises(LlmError, match="non-JSON"):
        await provider.complete(
            model="Qwen3.5-397B-A17B",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )


async def test_ovh_provider_guards_malformed_shape(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "boom"})

    settings = settings.model_copy(
        update={"ovh_api_key": "key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, _ = make_ovh_provider(settings, handler)
    with pytest.raises(LlmError, match="shape"):
        await provider.complete(
            model="Qwen3.5-397B-A17B",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )


async def test_ovh_provider_retries_on_500(settings: Settings) -> None:
    responses = [httpx.Response(500, json={}), ovh_ok_response()]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    settings = settings.model_copy(
        update={"ovh_api_key": "key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, _ = make_ovh_provider(settings, handler)
    completion = await provider.complete(
        model="Qwen3.5-397B-A17B",
        messages=[LlmMessage(role="user", content="hi")],
        thinking=True,
        json_mode=False,
        max_tokens=100,
        temperature=None,
        reasoning_effort=None,
    )
    assert completion.content == "ok"


async def test_ovh_provider_raises_on_401(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    settings = settings.model_copy(
        update={"ovh_api_key": "bad-key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    provider, _ = make_ovh_provider(settings, handler)
    with pytest.raises(LlmError, match="401"):
        await provider.complete(
            model="Qwen3.5-397B-A17B",
            messages=[LlmMessage(role="user", content="hi")],
            thinking=True,
            json_mode=False,
            max_tokens=100,
            temperature=None,
            reasoning_effort=None,
        )


async def test_ovh_provider_registry_entry(settings: Settings) -> None:
    from open_endurance_coach.clients.providers import build_registry

    settings = settings.model_copy(
        update={"ovh_api_key": "key", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    registry = build_registry(settings)
    assert "ovh" in registry
    assert isinstance(registry["ovh"], OvhProvider)


def _two_provider_client(
    settings: Settings,
) -> tuple[LlmClient, FakeLlmProvider, FakeLlmProvider]:
    fake = FakeLlmProvider([completion("fake-ok")])
    deepseek = FakeLlmProvider([completion("deepseek-ok")])
    client = LlmClient(
        settings.model_copy(update={"llm_provider": "fake", "llm_model": "fake-model"}),
        {"fake": fake, "deepseek": deepseek},
    )
    return client, fake, deepseek


def test_llm_client_reports_current_selection(settings: Settings) -> None:
    client, _, _ = _two_provider_client(settings)
    assert client.provider_name == "fake"
    assert client.model_name == "fake-model"


async def test_llm_client_select_switches_provider_and_default_model(settings: Settings) -> None:
    client, fake, deepseek = _two_provider_client(settings)
    client.select(provider="deepseek")
    assert client.provider_name == "deepseek"
    assert client.model_name == "deepseek-flash"
    result = await client.complete([LlmMessage(role="user", content="hi")])
    assert result.content == "deepseek-ok"
    assert deepseek.calls[0]["model"] == "deepseek-flash"
    assert fake.calls == []


def test_llm_client_select_model_only_keeps_provider(settings: Settings) -> None:
    client, _, _ = _two_provider_client(settings)
    client.select(model="custom-model")
    assert client.provider_name == "fake"
    assert client.model_name == "custom-model"


def test_llm_client_select_unknown_provider_raises(settings: Settings) -> None:
    client, _, _ = _two_provider_client(settings)
    with pytest.raises(LlmError, match="Unknown LLM provider"):
        client.select(provider="nope")


def test_llm_client_select_deepseek_without_key_raises(settings: Settings) -> None:
    keyless = settings.model_copy(update={"deepseek_api_key": ""})
    client = LlmClient(
        keyless.model_copy(update={"llm_provider": "fake", "llm_model": "fake-model"}),
        {"fake": FakeLlmProvider(), "deepseek": FakeLlmProvider()},
    )
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        client.select(provider="deepseek")


def test_llm_client_unknown_provider_reports_missing_key(settings: Settings) -> None:
    keyless = settings.model_copy(
        update={"deepseek_api_key": "", "llm_provider": "ovh", "llm_model": "Qwen3.5-397B-A17B"}
    )
    client = LlmClient(keyless, {"ovh": FakeLlmProvider(), "deepseek": FakeLlmProvider()})
    with pytest.raises(LlmError) as excinfo:
        client.select(provider="ova")
    message = str(excinfo.value)
    assert "not ready (needs DEEPSEEK_API_KEY)" in message
    assert "ready (no API key needed)" in message


def test_llm_client_unknown_provider_reports_key_present(settings: Settings) -> None:
    client, _, _ = _two_provider_client(settings)
    with pytest.raises(LlmError) as excinfo:
        client.select(provider="ova")
    assert "deepseek  ready (API key set)" in str(excinfo.value)

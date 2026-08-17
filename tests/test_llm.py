import pytest

from open_endurance_coach.clients.llm import LlmClient, LlmError, LlmMessage
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
    assert provider.calls[0]["model"] == "deepseek-v4-pro"


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

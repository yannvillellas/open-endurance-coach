import pytest
from pydantic import ValidationError

from open_endurance_coach.config import Settings


def test_defaults(settings: Settings) -> None:
    assert settings.app_timezone == "Europe/Paris"
    assert settings.llm_provider == "deepseek"
    assert settings.llm_model == "deepseek-v4-pro"
    assert settings.llm_thinking is True
    assert settings.intervals_athlete_id == "12345"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_TIMEZONE", "UTC")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    settings = Settings(intervals_api_key="k", deepseek_api_key="k")
    assert settings.app_timezone == "UTC"
    assert settings.llm_model == "deepseek-chat"


def test_invalid_timezone_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            intervals_api_key="k",
            deepseek_api_key="k",
            app_timezone="Not/AZone",
        )


def test_missing_secrets_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERVALS_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_chat_history_defaults(settings: Settings) -> None:
    assert settings.chat_history_turns == 10
    assert settings.chat_history_max_tokens == 2048


def test_chat_history_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_HISTORY_TURNS", "20")
    monkeypatch.setenv("CHAT_HISTORY_MAX_TOKENS", "4096")
    settings = Settings(intervals_api_key="k", deepseek_api_key="k")
    assert settings.chat_history_turns == 20
    assert settings.chat_history_max_tokens == 4096


def test_chat_history_settings_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(intervals_api_key="k", deepseek_api_key="k", chat_history_turns=0)
    with pytest.raises(ValidationError):
        Settings(intervals_api_key="k", deepseek_api_key="k", chat_history_max_tokens=0)
    with pytest.raises(ValidationError):
        Settings(intervals_api_key="k", deepseek_api_key="k", chat_history_max_age_days=0)


def test_chat_history_max_age_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(intervals_api_key="k", deepseek_api_key="k")
    assert settings.chat_history_max_age_days == 90
    monkeypatch.setenv("CHAT_HISTORY_MAX_AGE_DAYS", "30")
    settings = Settings(intervals_api_key="k", deepseek_api_key="k")
    assert settings.chat_history_max_age_days == 30

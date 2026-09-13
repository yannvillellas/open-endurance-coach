import pytest
from pydantic import ValidationError

from open_endurance_coach.config import PROVIDER_DEFAULT_MODELS, Settings


def test_defaults(settings: Settings) -> None:
    assert settings.app_timezone == "Europe/Paris"
    assert settings.llm_thinking is True
    assert settings.intervals_athlete_id == "12345"


def test_llm_defaults_are_ovh() -> None:
    settings = Settings(intervals_api_key="k", _env_file=None)
    assert settings.llm_provider == "ovh"
    assert settings.llm_model == "Qwen3.5-397B-A17B"


def test_provider_default_models_match_settings_defaults() -> None:
    settings = Settings(intervals_api_key="k", _env_file=None)
    assert PROVIDER_DEFAULT_MODELS[settings.llm_provider] == settings.llm_model


def test_provider_selected_by_env_resolves_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = Settings(intervals_api_key="k", deepseek_api_key="k", _env_file=None)
    assert settings.llm_provider == "deepseek"
    assert settings.llm_model == "deepseek-flash"


def test_deepseek_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="DEEPSEEK_API_KEY"):
        Settings(intervals_api_key="k", llm_provider="deepseek", _env_file=None)


def test_with_llm_override_rejects_deepseek_without_key() -> None:
    settings = Settings(intervals_api_key="k", _env_file=None)
    assert settings.llm_provider == "ovh"
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        settings.with_llm_override(provider="deepseek")


def test_llm_keys_are_optional() -> None:
    settings = Settings(intervals_api_key="k", _env_file=None)
    assert settings.ovh_api_key == ""
    assert settings.deepseek_api_key == ""


def test_with_llm_override_switches_provider_and_default_model(settings: Settings) -> None:
    overridden = settings.with_llm_override(provider="deepseek")
    assert overridden.llm_provider == "deepseek"
    assert overridden.llm_model == "deepseek-flash"


def test_with_llm_override_ovh_default_model(settings: Settings) -> None:
    overridden = settings.with_llm_override(provider="ovh")
    assert overridden.llm_provider == "ovh"
    assert overridden.llm_model == "Qwen3.5-397B-A17B"


def test_with_llm_override_explicit_model_wins(settings: Settings) -> None:
    overridden = settings.with_llm_override(provider="deepseek", model="deepseek-chat")
    assert overridden.llm_model == "deepseek-chat"


def test_with_llm_override_model_only_keeps_provider(settings: Settings) -> None:
    overridden = settings.with_llm_override(model="custom-model")
    assert overridden.llm_provider == settings.llm_provider
    assert overridden.llm_model == "custom-model"


def test_with_llm_override_noop_returns_same_settings(settings: Settings) -> None:
    assert settings.with_llm_override() is settings


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


def test_missing_intervals_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERVALS_API_KEY", raising=False)
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

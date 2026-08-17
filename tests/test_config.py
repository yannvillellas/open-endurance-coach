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

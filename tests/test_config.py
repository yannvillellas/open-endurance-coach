import pytest
from pydantic import ValidationError

from open_endurance_coach.config import (
    PROVIDER_DEFAULT_MODELS,
    PROVIDERS,
    Settings,
    effective_input_budget,
    resolved_context_window,
    resolved_input_room,
    resolved_max_output_tokens,
    validate_llm_limits,
)
from open_endurance_coach.tokens import BUDGET_SAFETY_MARGIN


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
    settings = Settings(intervals_api_key="k", deepseek_api_key="k", _env_file=None)
    assert settings.app_timezone == "UTC"
    assert settings.llm_model == "deepseek-chat"


def test_invalid_timezone_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            intervals_api_key="k",
            deepseek_api_key="k",
            app_timezone="Not/AZone",
            _env_file=None,
        )


def test_missing_intervals_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERVALS_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_chat_history_defaults(settings: Settings) -> None:
    assert settings.chat_history_turns == 10


def test_chat_history_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_HISTORY_TURNS", "20")
    settings = Settings(intervals_api_key="k", deepseek_api_key="k", _env_file=None)
    assert settings.chat_history_turns == 20


def test_chat_history_settings_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(intervals_api_key="k", deepseek_api_key="k", chat_history_turns=0, _env_file=None)
    with pytest.raises(ValidationError):
        Settings(
            intervals_api_key="k",
            deepseek_api_key="k",
            chat_history_max_age_days=0,
            _env_file=None,
        )


def test_chat_history_max_age_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(intervals_api_key="k", deepseek_api_key="k", _env_file=None)
    assert settings.chat_history_max_age_days == 90
    monkeypatch.setenv("CHAT_HISTORY_MAX_AGE_DAYS", "30")
    settings = Settings(intervals_api_key="k", deepseek_api_key="k", _env_file=None)
    assert settings.chat_history_max_age_days == 30


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown LLM provider"):
        Settings(intervals_api_key="k", llm_provider="bogus", _env_file=None)


@pytest.mark.parametrize("value", ["12345/../admin", "a b", "i" * 40, ""])
def test_athlete_id_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValidationError, match="intervals_athlete_id"):
        Settings(intervals_api_key="k", intervals_athlete_id=value, _env_file=None)


def test_athlete_id_accepts_supported_forms() -> None:
    explicit = Settings(intervals_api_key="k", intervals_athlete_id="i12345", _env_file=None)
    assert explicit.intervals_athlete_id == "i12345"
    padded = Settings(intervals_api_key="k", intervals_athlete_id=" 0 ", _env_file=None)
    assert padded.intervals_athlete_id == "0"


def test_provider_specs_declare_model_limits() -> None:
    for spec in PROVIDERS.values():
        assert spec.default_model
        assert spec.context_window > 0
        assert spec.max_output_tokens > 0


def test_resolved_context_window_from_provider_defaults() -> None:
    ovh = Settings(intervals_api_key="k", _env_file=None)
    assert resolved_context_window(ovh) == 262144
    deepseek = Settings(
        intervals_api_key="k", deepseek_api_key="k", llm_provider="deepseek", _env_file=None
    )
    assert resolved_context_window(deepseek) == 1048576


def test_resolved_context_window_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "1000")
    settings = Settings(intervals_api_key="k", _env_file=None)
    assert resolved_context_window(settings) == 1000


def test_unknown_model_requires_context_window() -> None:
    settings = Settings(intervals_api_key="k", llm_model="custom-model", _env_file=None)
    with pytest.raises(ValueError, match="LLM_CONTEXT_WINDOW"):
        resolved_context_window(settings)


def test_unknown_model_with_context_window_resolves() -> None:
    settings = Settings(
        intervals_api_key="k", llm_model="custom-model", llm_context_window=200000, _env_file=None
    )
    assert resolved_context_window(settings) == 200000


def test_resolved_max_output_tokens_from_registry_and_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(intervals_api_key="k", _env_file=None)
    assert resolved_max_output_tokens(settings) == 262144
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "9000")
    assert resolved_max_output_tokens(Settings(intervals_api_key="k", _env_file=None)) == 9000


def test_unknown_model_without_output_override_keeps_reply_cap() -> None:
    settings = Settings(intervals_api_key="k", llm_model="custom-model", _env_file=None)
    assert resolved_max_output_tokens(settings) == settings.llm_max_tokens


def test_validate_llm_limits_rejects_reply_above_output_cap() -> None:
    settings = Settings(
        intervals_api_key="k",
        llm_model="custom-model",
        llm_context_window=200000,
        llm_max_output_tokens=8192,
        llm_max_tokens=9000,
        _env_file=None,
    )
    with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
        validate_llm_limits(settings)


def test_validate_llm_limits_accepts_defaults() -> None:
    validate_llm_limits(Settings(intervals_api_key="k", _env_file=None))


def test_llm_input_budget_is_unset_by_default_and_env_settable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings(intervals_api_key="k", _env_file=None).llm_input_budget is None
    monkeypatch.setenv("LLM_INPUT_BUDGET", "12345")
    assert Settings(intervals_api_key="k", _env_file=None).llm_input_budget == 12345


def test_budget_safety_margin_is_a_tunable_constant() -> None:
    assert BUDGET_SAFETY_MARGIN == 0.2


def test_effective_input_budget_defaults_to_the_model_window() -> None:
    ovh = Settings(intervals_api_key="k", _env_file=None)
    assert effective_input_budget(ovh) == 176948
    deepseek = Settings(
        intervals_api_key="k", deepseek_api_key="k", llm_provider="deepseek", _env_file=None
    )
    assert effective_input_budget(deepseek) == 806093


def test_resolved_input_room_ignores_the_soft_cap() -> None:
    settings = Settings(intervals_api_key="k", llm_input_budget=500_000, _env_file=None)
    assert resolved_input_room(settings) == 176948
    assert effective_input_budget(settings) == 176948


def test_effective_input_budget_caps_below_the_window() -> None:
    ovh = Settings(intervals_api_key="k", llm_input_budget=500_000, _env_file=None)
    assert effective_input_budget(ovh) == 176948
    deepseek = Settings(
        intervals_api_key="k",
        deepseek_api_key="k",
        llm_provider="deepseek",
        llm_input_budget=500_000,
        _env_file=None,
    )
    assert effective_input_budget(deepseek) == 500_000


def test_effective_input_budget_rejects_a_reply_cap_above_the_model_output() -> None:
    settings = Settings(
        intervals_api_key="k",
        deepseek_api_key="k",
        llm_provider="deepseek",
        llm_max_tokens=400_000,
        _env_file=None,
    )
    with pytest.raises(ValueError, match="exceeds the output cap"):
        effective_input_budget(settings)


def test_too_small_window_has_no_input_room() -> None:
    settings = Settings(
        intervals_api_key="k", llm_context_window=10_000, llm_max_tokens=32768, _env_file=None
    )
    with pytest.raises(ValueError, match="no input room"):
        effective_input_budget(settings)


def test_settings_errors_do_not_echo_input_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERVALS_API_KEY", "FAKE-SECRET-VALUE")
    monkeypatch.setenv("LLM_MAX_TOKENS", "not-a-number")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "FAKE-SECRET-VALUE" not in str(excinfo.value)

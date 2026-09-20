import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ATHLETE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")


@dataclass(frozen=True)
class ProviderSpec:
    default_model: str
    api_key_env: str | None


PROVIDERS: dict[str, ProviderSpec] = {
    "ovh": ProviderSpec(default_model="Qwen3.5-397B-A17B", api_key_env=None),
    "deepseek": ProviderSpec(default_model="deepseek-flash", api_key_env="DEEPSEEK_API_KEY"),
}

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    name: spec.default_model for name, spec in PROVIDERS.items()
}
PROVIDER_API_KEY_ENV: dict[str, str | None] = {
    name: spec.api_key_env for name, spec in PROVIDERS.items()
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    intervals_api_key: str
    intervals_athlete_id: str = "0"
    intervals_base_url: str = "https://intervals.icu/api/v1"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    ovh_api_key: str = ""
    ovh_base_url: str = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"

    llm_provider: str = "ovh"
    llm_model: str = ""
    llm_thinking: bool = True
    llm_reasoning_effort: str | None = None
    llm_max_tokens: int = 32768
    llm_temperature: float | None = None
    llm_timeout_seconds: float = 180.0

    # Training-day boundaries and calendar dates are written as midnight in this zone;
    # it must match the athlete's Intervals.icu account timezone.
    app_timezone: str = "Europe/Paris"
    database_path: str = "data/coach.db"

    athlete_profile: str = ""
    coach_tone: str = "Be objective, strict, and analytical. Do not offer generic encouragement."

    chat_history_turns: int = Field(default=10, ge=1)
    chat_history_max_tokens: int = Field(default=16384, ge=1)
    chat_history_max_age_days: int = Field(default=90, ge=1)
    history_days: int = Field(default=180, ge=0)

    requests_per_second: float = 8.0
    max_retries: int = 3
    retry_base_delay: float = 1.0

    @field_validator("app_timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {value}") from exc
        return value

    @field_validator("intervals_athlete_id")
    @classmethod
    def _validate_athlete_id(cls, value: str) -> str:
        stripped = value.strip()
        if not _ATHLETE_ID_RE.fullmatch(stripped):
            raise ValueError("intervals_athlete_id must be a safe id such as 'i12345' or '0'")
        return stripped

    @model_validator(mode="after")
    def _resolve_llm_defaults(self) -> "Settings":
        if self.llm_provider not in PROVIDERS:
            raise ValueError(
                f"Unknown LLM provider: {self.llm_provider!r}\nAvailable providers:\n"
                f"{describe_providers(PROVIDERS, self)}"
            )
        if not self.llm_model:
            self.llm_model = PROVIDER_DEFAULT_MODELS[self.llm_provider]
        self._check_provider_credentials(self.llm_provider)
        return self

    def _check_provider_credentials(self, provider: str) -> None:
        spec = PROVIDERS.get(provider)
        if spec is None:
            return
        if spec.api_key_env is not None and not getattr(self, spec.api_key_env.lower()):
            raise ValueError(f"No API key for provider {provider!r}; set {spec.api_key_env}")

    def with_llm_override(
        self, *, provider: str | None = None, model: str | None = None
    ) -> "Settings":
        if provider is None and model is None:
            return self
        resolved_provider = provider or self.llm_provider
        self._check_provider_credentials(resolved_provider)
        resolved_model = model or PROVIDER_DEFAULT_MODELS.get(resolved_provider, self.llm_model)
        return self.model_copy(
            update={"llm_provider": resolved_provider, "llm_model": resolved_model}
        )


def describe_providers(names: Iterable[str], settings: Settings) -> str:
    rows: list[tuple[str, str]] = []
    for name in sorted(names):
        key_env = PROVIDER_API_KEY_ENV.get(name)
        if key_env is None:
            status = "ready (no API key needed)"
        elif getattr(settings, key_env.lower(), ""):
            status = "ready (API key set)"
        else:
            status = f"not ready (needs {key_env})"
        rows.append((name, status))
    if not rows:
        return "  (none configured)"
    width = max(len(name) for name, _ in rows)
    return "\n".join(f"  {name:<{width}}  {status}" for name, status in rows)


@lru_cache
def get_settings() -> Settings:
    return Settings()

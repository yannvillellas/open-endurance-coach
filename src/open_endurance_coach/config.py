from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    intervals_api_key: str
    intervals_athlete_id: str = "0"
    intervals_base_url: str = "https://intervals.icu/api/v1"

    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"

    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-v4-pro"
    llm_thinking: bool = True
    llm_reasoning_effort: str | None = None
    llm_max_tokens: int = 8192
    llm_temperature: float | None = None
    llm_timeout_seconds: float = 180.0

    app_timezone: str = "Europe/Paris"
    database_path: str = "data/coach.db"
    http_port: int = 8000

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


@lru_cache
def get_settings() -> Settings:
    return Settings()

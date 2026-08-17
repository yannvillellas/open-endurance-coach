import pytest

from open_endurance_coach.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        intervals_api_key="test-intervals-key",
        intervals_athlete_id="12345",
        deepseek_api_key="test-llm-key",
        requests_per_second=100000.0,
        retry_base_delay=0.0,
    )

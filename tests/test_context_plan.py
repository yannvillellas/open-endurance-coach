from datetime import date

import pytest

from open_endurance_coach.config import Settings
from open_endurance_coach.extractors import context as context_module
from open_endurance_coach.extractors import deep as deep_module
from open_endurance_coach.extractors.context import ContextPlan, StandardExtractor
from open_endurance_coach.extractors.deep import DeepHistoricalExtractor, detect_deep_query
from open_endurance_coach.extractors.standard import (
    ACTIVITY_LOOKBACK_DAYS,
    RACE_HORIZON_DAYS,
    RECENT_EVENT_DAYS,
    ROLLUP_LOOKBACK_DAYS,
    UPCOMING_DAYS,
    WELLNESS_LOOKBACK_DAYS,
)

from .fakes import make_intervals_client

TODAY = date(2024, 2, 1)


def test_standard_plan_matches_the_tactical_windows() -> None:
    plan = ContextPlan.standard()

    assert plan.activities_days == ACTIVITY_LOOKBACK_DAYS
    assert plan.wellness_days == WELLNESS_LOOKBACK_DAYS
    assert plan.event_past_days == RECENT_EVENT_DAYS
    assert plan.event_future_days == UPCOMING_DAYS
    assert plan.race_horizon_days == RACE_HORIZON_DAYS
    assert plan.rollup_days == ROLLUP_LOOKBACK_DAYS
    assert plan.activity_types == frozenset()
    assert plan.metric_focus is None
    assert plan.reference is None
    assert plan.detail is False


def test_for_activities_overrides_only_the_activity_selection() -> None:
    plan = ContextPlan.standard().for_activities(
        days=90,
        activity_types=frozenset({"Ride"}),
        metric_focus="heart_rate",
        reference=date(2024, 1, 15),
        detail=True,
    )

    assert plan.activities_days == 90
    assert plan.activity_types == frozenset({"Ride"})
    assert plan.metric_focus == "heart_rate"
    assert plan.reference == date(2024, 1, 15)
    assert plan.detail is True
    assert plan.wellness_days == WELLNESS_LOOKBACK_DAYS
    assert plan.event_past_days == RECENT_EVENT_DAYS
    assert plan.race_horizon_days == RACE_HORIZON_DAYS
    assert plan.rollup_days == ROLLUP_LOOKBACK_DAYS


async def test_both_extractors_feed_the_single_gather(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans: list[ContextPlan] = []

    async def fake_gather(
        client: object, plan: ContextPlan, focus: str, **kwargs: object
    ) -> object:
        plans.append(plan)
        return None

    monkeypatch.setattr(context_module, "gather_context", fake_gather)
    monkeypatch.setattr(deep_module, "gather_context", fake_gather)
    client = make_intervals_client()

    await StandardExtractor(settings, client).extract("status check", today=TODAY)

    focus = "how has my heart rate trend over the last 3 months"
    query = detect_deep_query(focus, today=TODAY)
    assert query is not None
    await DeepHistoricalExtractor(settings, client).extract(focus, query=query, today=TODAY)

    assert plans[0] == ContextPlan.standard()
    assert plans[1].activities_days == 90
    assert plans[1].metric_focus == "heart_rate"
    assert plans[1].detail is True

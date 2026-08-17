import argparse
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from open_endurance_coach.clients.intervals import IntervalsClient
from open_endurance_coach.config import Settings
from open_endurance_coach.fixtures.anonymize import anonymize_fixtures

ACTIVITY_WINDOW_DAYS = 30
EVENT_PAST_DAYS = 7
EVENT_FUTURE_DAYS = 60

# The coaching domain is cycling-centric, so the detail fixture should
# prefer a ride (rich power/HR intervals) over the latest activity overall.
PREFERRED_DETAIL_TYPES = (
    "Ride",
    "VirtualRide",
    "GravelRide",
    "TrackRide",
    "Cyclocross",
    "MountainBikeRide",
)

FIXTURE_FILENAMES = (
    "activities.json",
    "activity_detail.json",
    "wellness.json",
    "events.json",
    "sport_settings.json",
    "athlete_summary.json",
)


class FixtureSource(Protocol):
    async def list_activities(self, oldest: str, newest: str) -> list[dict[str, Any]]: ...
    async def get_activity(self, activity_id: str, intervals: bool = True) -> dict[str, Any]: ...
    async def list_wellness(self, oldest: str, newest: str) -> list[dict[str, Any]]: ...
    async def list_events(self, oldest: str, newest: str) -> list[dict[str, Any]]: ...
    async def get_sport_settings(self) -> list[dict[str, Any]]: ...
    async def get_athlete_summary(self) -> dict[str, Any]: ...


async def record_fixtures(settings: Settings, client: FixtureSource) -> dict[str, Any]:
    timezone = ZoneInfo(settings.app_timezone)
    today = datetime.now(timezone).date()
    activities_oldest = (today - timedelta(days=ACTIVITY_WINDOW_DAYS)).isoformat()
    activities_newest = (today + timedelta(days=1)).isoformat()

    activities = await client.list_activities(activities_oldest, activities_newest)
    if not activities:
        raise RuntimeError(
            f"no activities in window {activities_oldest}..{activities_newest}; "
            "record fixtures only with real athlete data present"
        )
    rides = [item for item in activities if item.get("type") in PREFERRED_DETAIL_TYPES]
    latest = max(rides or activities, key=lambda item: str(item.get("start_date_local", "")))
    detail = await client.get_activity(str(latest["id"]), intervals=True)
    wellness = await client.list_wellness(activities_oldest, activities_newest)
    events = await client.list_events(
        (today - timedelta(days=EVENT_PAST_DAYS)).isoformat(),
        (today + timedelta(days=EVENT_FUTURE_DAYS)).isoformat(),
    )
    sport_settings = await client.get_sport_settings()
    athlete_summary = await client.get_athlete_summary()

    return anonymize_fixtures(
        {
            "activities.json": activities,
            "activity_detail.json": detail,
            "wellness.json": wellness,
            "events.json": events,
            "sport_settings.json": sport_settings,
            "athlete_summary.json": athlete_summary,
        }
    )


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "tests" / "fixtures"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record read-only Intervals.icu payloads as anonymized test fixtures"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_output_dir(),
        help="output directory (default: tests/fixtures)",
    )
    args = parser.parse_args()

    settings = Settings()
    output_dir: Path = args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    async def run() -> None:
        client = IntervalsClient(settings)
        try:
            fixtures = await record_fixtures(settings, client)
        finally:
            await client.aclose()
        for filename in FIXTURE_FILENAMES:
            path = output_dir / filename
            with path.open("w", encoding="utf-8") as handle:
                json.dump(fixtures[filename], handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            print(f"wrote {path}")

    asyncio.run(run())


if __name__ == "__main__":
    main()

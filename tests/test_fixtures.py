import json
import re
from pathlib import Path
from typing import Any

from open_endurance_coach.fixtures.anonymize import (
    COORD_KEYS,
    EPOCH_END,
    EPOCH_START,
    FREE_TEXT_KEYS,
    GENERATED_NAME_CONTEXTS,
    SYNTHETIC_VOCABULARY,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"

FIXTURE_SPEC: dict[str, type] = {
    "activities.json": list,
    "activity_detail.json": dict,
    "wellness.json": list,
    "events.json": list,
    "sport_settings.json": list,
    "athlete_summary.json": list,
}

# The athlete's Intervals.icu calendar contains no events, so the recorded
# events fixture is legitimately empty; all other fixtures must be non-empty.
MAY_BE_EMPTY = {"events.json"}

REAL_ID_RE = re.compile(r"^[a-z]\d{5,}$")
SYNTHETIC_ID_RE = re.compile(r"^fx\d{6}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2})?")


def _is_id_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered == "id"
        or lowered.endswith("_id")
        or lowered.endswith("_ids")
        or lowered.startswith("id_")
    )


def _is_iso_date_string(value: str) -> bool:
    return bool(ISO_DATE_RE.match(value))


def load_fixtures() -> dict[str, Any]:
    fixtures: dict[str, Any] = {}
    for filename in FIXTURE_SPEC:
        path = FIXTURE_DIR / filename
        assert path.exists(), f"missing fixture file: {filename}"
        with path.open(encoding="utf-8") as handle:
            fixtures[filename] = json.load(handle)
    return fixtures


def walk(fixtures: dict[str, Any]) -> list[tuple[str, str, Any, str]]:
    found: list[tuple[str, str, Any, str]] = []

    def visit(value: Any, key: str = "", parent_key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key), parent_key=key)
        elif isinstance(value, list):
            for item in value:
                visit(item, key, parent_key=parent_key)
        else:
            found.append((key, type(value).__name__, value, parent_key))

    for payload in fixtures.values():
        visit(payload)
    return found


def test_fixtures_are_valid_and_anonymized() -> None:
    fixtures = load_fixtures()
    for filename, expected_type in FIXTURE_SPEC.items():
        assert isinstance(fixtures[filename], expected_type), (
            f"{filename}: expected {expected_type.__name__}"
        )
        if filename not in MAY_BE_EMPTY:
            assert fixtures[filename], f"{filename}: fixture is empty"


def test_fixtures_contain_no_real_ids() -> None:
    for key, _kind, value, _parent in walk(load_fixtures()):
        if isinstance(value, str):
            assert not REAL_ID_RE.match(value), f"{key}: real-looking id leaked: {value}"
            if _is_id_key(key) and not _is_iso_date_string(value):
                assert SYNTHETIC_ID_RE.match(value), f"{key}: id not synthetic: {value}"
        elif isinstance(value, int) and not isinstance(value, bool):
            if _is_id_key(key) and value != 0:
                assert 10000 <= value < 20000, f"{key}: real numeric id leaked: {value}"


def test_all_dates_are_in_synthetic_window() -> None:
    for key, _, value, _parent in walk(load_fixtures()):
        if isinstance(value, str) and _is_iso_date_string(value):
            assert value.startswith("2024"), f"{key}: date outside synthetic window: {value}"
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and 1e9 <= value < 3e9:
            assert EPOCH_START <= value < EPOCH_END, f"{key}: real epoch timestamp leaked: {value}"


def test_no_emails_urls_or_long_free_text() -> None:
    for key, _, value, parent_key in walk(load_fixtures()):
        if isinstance(value, str):
            assert "@" not in value, f"{key}: email leaked: {value}"
            assert not value.startswith(("http://", "https://")), f"{key}: url leaked: {value}"
            assert len(value) <= 120, f"{key}: long free text leaked ({len(value)} chars)"
            if key.lower() in FREE_TEXT_KEYS and not (
                key.lower() == "name" and parent_key in GENERATED_NAME_CONTEXTS
            ):
                assert value in SYNTHETIC_VOCABULARY, (
                    f"{key}: non-synthetic free text leaked: {value}"
                )


def test_coordinates_are_obfuscated() -> None:
    for key, _, value, _parent in walk(load_fixtures()):
        if (
            key.lower() not in COORD_KEYS
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            continue
        assert round(value, 2) == value, f"{key}: coordinate precision leaked: {value}"
        if key.lower().endswith(("lat", "latitude")):
            assert -90.0 <= value <= 90.0
        elif key.lower().endswith(("lng", "lon", "longitude")):
            assert -180.0 <= value <= 180.0


def test_activity_detail_id_appears_in_activity_list() -> None:
    fixtures = load_fixtures()
    detail_id = fixtures["activity_detail.json"]["id"]
    listed_ids = {str(item["id"]) for item in fixtures["activities.json"]}
    assert detail_id in listed_ids, "activity_detail id not present in activities list"

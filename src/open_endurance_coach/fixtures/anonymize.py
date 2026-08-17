import hashlib
import re
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

SYNTHETIC_WINDOW_TZ = ZoneInfo("Europe/Paris")
WINDOW_START = datetime(2024, 1, 1, tzinfo=SYNTHETIC_WINDOW_TZ)
WINDOW_END = datetime(2025, 1, 1, tzinfo=SYNTHETIC_WINDOW_TZ)
EPOCH_START = int(WINDOW_START.timestamp())
EPOCH_END = int(WINDOW_END.timestamp())

SYNTHETIC_VOCABULARY: tuple[str, ...] = (
    "Endurance Block",
    "Tempo Session",
    "Recovery Spin",
    "Threshold Intervals",
    "Criterium Skills",
    "Long Ride",
    "Open Water Swim",
    "Trail Run",
    "Strength Routine",
    "Rest Day",
    "Fixture Athlete",
    "Synthetic Workout",
)

FREE_TEXT_KEYS = frozenset(
    {
        "name",
        "names",
        "description",
        "note",
        "notes",
        "comment",
        "comments",
        "title",
        "summary",
        "first_name",
        "last_name",
        "username",
        "location",
        "address",
    }
)

COORD_KEYS = frozenset(
    {
        "lat",
        "lng",
        "lon",
        "latitude",
        "longitude",
        "latlng",
        "position",
        "coordinates",
        "start_latlng",
        "end_latlng",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
    }
)

JITTER_KEYS = frozenset({"weight", "weight_kg", "ftp"})

GENERATED_NAME_CONTEXTS = frozenset({"icu_intervals"})

LAT_OFFSET = 1.3721
LNG_OFFSET = -2.3412

ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$"
)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
COORD_STRING_RE = re.compile(r"^-?\d+(\.\d+)?(\s*,\s*-?\d+(\.\d+)?)?$")

_MIN_EPOCH = 1_000_000_000
_MAX_EPOCH = 3_000_000_000


def _is_id_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered == "id"
        or lowered in {"group", "groups"}
        or lowered.endswith("_id")
        or lowered.endswith("_ids")
        or lowered.startswith("id_")
    )


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace(" ", "T")
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.fromisoformat(normalized[:10])
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SYNTHETIC_WINDOW_TZ)
    return parsed


def _hash(value: Any, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    return int(digest[:16], 16)


class _Anonymizer:
    def __init__(self, payloads: list[Any]) -> None:
        self._id_map: dict[str, str] = {}
        self._id_counter = 0
        self._num_id_map: dict[int, int] = {}
        self._num_id_counter = 0
        self._min_ts: float | None = None
        self._max_ts: float | None = None
        for payload in payloads:
            self._collect_timestamps(payload)

    def _collect_timestamps(self, value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                self._collect_timestamps(child)
        elif isinstance(value, list):
            for item in value:
                self._collect_timestamps(item)
        elif isinstance(value, str):
            if ISO_DATETIME_RE.match(value) or ISO_DATE_RE.match(value):
                self._register(_parse_datetime(value).timestamp())
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and _MIN_EPOCH <= value < _MAX_EPOCH
        ):
            self._register(float(value))

    def _register(self, timestamp: float) -> None:
        if self._min_ts is None or timestamp < self._min_ts:
            self._min_ts = timestamp
        if self._max_ts is None or timestamp > self._max_ts:
            self._max_ts = timestamp

    def _remap_timestamp(self, timestamp: float) -> float:
        if self._min_ts is None or self._max_ts is None:
            return timestamp
        span = self._max_ts - self._min_ts
        offset = timestamp - self._min_ts if span > 0 else 0.0
        remapped = EPOCH_START + offset
        return min(max(remapped, float(EPOCH_START)), float(EPOCH_END - 1))

    def _remap_date(self, value: str) -> str:
        original = _parse_datetime(value)
        remapped = datetime.fromtimestamp(
            self._remap_timestamp(original.timestamp()), SYNTHETIC_WINDOW_TZ
        )
        if ISO_DATE_RE.match(value):
            return remapped.date().isoformat()
        if value.endswith("Z"):
            return remapped.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        return remapped.strftime("%Y-%m-%dT%H:%M:%S")

    def _synthetic_id(self, value: str) -> str:
        if value not in self._id_map:
            self._id_counter += 1
            self._id_map[value] = f"fx{self._id_counter:06d}"
        return self._id_map[value]

    def _synthetic_num_id(self, value: int) -> int:
        if value not in self._num_id_map:
            self._num_id_map[value] = 10000 + self._num_id_counter
            self._num_id_counter += 1
        return self._num_id_map[value]

    def _vocabulary(self, value: str) -> str:
        return SYNTHETIC_VOCABULARY[_hash(value, "vocab") % len(SYNTHETIC_VOCABULARY)]

    def _jitter(self, value: float) -> float:
        magnitude = (_hash(value, "jitter") % 19) + 1
        sign = 1.0 if (_hash(value, "jitter") >> 16) & 1 else -1.0
        return value * (1.0 + sign * magnitude / 1000.0)

    def _offset_coord(self, value: float, is_lat: bool) -> float:
        offset = LAT_OFFSET if is_lat else LNG_OFFSET
        shifted = value + offset
        limit = 90.0 if is_lat else 180.0
        while shifted > limit:
            shifted -= 2 * limit
        while shifted < -limit:
            shifted += 2 * limit
        return round(shifted, 2)

    def _coord_value(self, value: float, key: str, index: int | None) -> float:
        lowered = key.lower()
        is_lat = lowered.endswith(("lat", "latitude")) if index is None else index % 2 == 0
        return self._offset_coord(value, is_lat)

    def anonymize(
        self, value: Any, key: str = "", index: int | None = None, parent_key: str = ""
    ) -> Any:
        if isinstance(value, dict):
            return {
                child_key: self.anonymize(child_value, str(child_key), parent_key=key)
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [
                self.anonymize(
                    item,
                    key,
                    i if key.lower() in COORD_KEYS else None,
                    parent_key=parent_key,
                )
                for i, item in enumerate(value)
            ]
        if isinstance(value, str):
            return self._anonymize_string(value, key, index, parent_key)
        if isinstance(value, float):
            return self._anonymize_number(value, key, index)
        if isinstance(value, int) and not isinstance(value, bool):
            return self._anonymize_number(value, key, index)
        return value

    def _anonymize_string(self, value: str, key: str, index: int | None, parent_key: str) -> str:
        if ISO_DATETIME_RE.match(value) or ISO_DATE_RE.match(value):
            return self._remap_date(value)
        if _is_id_key(key):
            return self._synthetic_id(value)
        if "@" in value:
            return "fixture-email"
        if value.startswith(("http://", "https://")):
            return "fixture-url"
        lowered = key.lower()
        if lowered in COORD_KEYS and COORD_STRING_RE.match(value):
            parts = [float(part.strip()) for part in value.split(",")]
            is_lat = index is None or index % 2 == 0
            remapped = [self._offset_coord(parts[0], is_lat)]
            if len(parts) > 1:
                remapped.append(self._offset_coord(parts[1], not is_lat))
            return ", ".join(f"{part:.2f}" for part in remapped)
        if (
            lowered in FREE_TEXT_KEYS
            and not (lowered == "name" and parent_key in GENERATED_NAME_CONTEXTS)
        ) or len(value) > 120:
            return self._vocabulary(value)
        return value

    def _anonymize_number(self, value: int | float, key: str, index: int | None) -> int | float:
        lowered = key.lower()
        if _is_id_key(key):
            if value == 0:
                return int(value)
            return self._synthetic_num_id(int(value))
        if lowered in COORD_KEYS:
            return self._coord_value(float(value), key, index)
        if lowered in JITTER_KEYS:
            return round(self._jitter(float(value)), 4)
        if _MIN_EPOCH <= value < _MAX_EPOCH:
            return int(self._remap_timestamp(float(value)))
        if isinstance(value, float):
            return round(value, 4)
        return value


def anonymize_fixtures(fixtures: dict[str, Any]) -> dict[str, Any]:
    anonymizer = _Anonymizer(list(fixtures.values()))
    return {name: anonymizer.anonymize(payload) for name, payload in fixtures.items()}

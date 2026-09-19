import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from open_endurance_coach.schemas.intervals import ActivitySplit

STREAM_TYPES = ("time", "distance", "altitude", "heartrate")
RIDE_STREAM_TYPES = (*STREAM_TYPES, "watts")


def stream_types(speed_based: bool) -> tuple[str, ...]:
    """Streams to request: rides add watts, since power is their intensity metric."""
    return RIDE_STREAM_TYPES if speed_based else STREAM_TYPES


_KM = 1000.0
_MIN_TAIL_M = 100.0
MAX_SPLITS = 20


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _segment(
    *,
    label: str,
    times: Sequence[Any],
    distances: Sequence[Any],
    altitudes: Sequence[Any],
    heart_rates: Sequence[Any],
    watts: Sequence[Any],
    speed_based: bool,
    start: int,
    end: int,
) -> ActivitySplit | None:
    start_time = _number(times[start]) if start < len(times) else None
    end_time = _number(times[end]) if end < len(times) else None
    start_distance = _number(distances[start])
    end_distance = _number(distances[end])
    if start_time is None or end_time is None or start_distance is None or end_distance is None:
        return None
    distance = end_distance - start_distance
    if distance <= 0:
        return None
    seconds = round(end_time - start_time)
    if seconds <= 0:
        return None

    power = [value for value in (_number(v) for v in watts[start : end + 1]) if value is not None]
    rates = [
        rate
        for rate in (_number(value) for value in heart_rates[start : end + 1])
        if rate is not None
    ]
    heights = [
        height
        for height in (_number(value) for value in altitudes[start : end + 1])
        if height is not None
    ]
    gain = sum(max(0.0, later - earlier) for earlier, later in pairwise(heights))
    loss = sum(max(0.0, earlier - later) for earlier, later in pairwise(heights))
    grade = (heights[-1] - heights[0]) / distance * 100 if len(heights) >= 2 else None

    return ActivitySplit(
        label=label,
        distance_m=round(distance, 1),
        time_s=seconds,
        pace_s_per_km=None if speed_based else round(seconds / (distance / _KM)),
        average_speed_kmh=(round((distance / _KM) / (seconds / 3600), 1) if speed_based else None),
        average_watts=round(sum(power) / len(power)) if power else None,
        average_heartrate=round(sum(rates) / len(rates)) if rates else None,
        max_heartrate=round(max(rates)) if rates else None,
        elevation_gain_m=round(gain, 1),
        elevation_loss_m=round(loss, 1),
        grade_pct=round(grade, 1) if grade is not None else None,
    )


def _chunk_label(number: int, chunk_km: int, *, final: bool = False) -> str:
    if chunk_km <= 1:
        return f"km {number}" + (" (partial)" if final else "")
    first = (number - 1) * chunk_km + 1
    if final:
        return f"km {first}-end"
    return f"km {first}-{number * chunk_km}"


def per_km_splits(
    streams: Mapping[str, Sequence[Any]],
    *,
    speed_based: bool = False,
    max_splits: int = MAX_SPLITS,
) -> list[ActivitySplit]:
    """Splits computed from streams, capped so a long activity stays readable.

    One row per kilometre, plus a trailing partial; when that would exceed
    ``max_splits`` rows (a 150 km ride, say) consecutive kilometres are merged so the
    table stays bounded. Read-only: nothing is written to Intervals and no intervals
    need to exist on the activity. Returns [] without a usable time or distance series.
    """
    times = streams.get("time") or []
    distances = streams.get("distance") or []
    altitudes = streams.get("altitude") or []
    heart_rates = streams.get("heartrate") or []
    watts = streams.get("watts") or []
    if not times or not distances:
        return []

    count = min(len(times), len(distances))
    total = _number(distances[count - 1]) or 0.0
    if total <= 0:
        return []
    chunk_km = 1
    if max_splits > 0 and total > max_splits * _KM:
        chunk_km = math.ceil(total / _KM / max_splits)
    chunk_m = chunk_km * _KM

    splits: list[ActivitySplit] = []
    start = 0
    boundary = chunk_m
    number = 1
    for index in range(count):
        distance = _number(distances[index])
        if distance is None or distance < boundary:
            continue
        split = _segment(
            label=_chunk_label(number, chunk_km),
            times=times,
            distances=distances,
            altitudes=altitudes,
            heart_rates=heart_rates,
            watts=watts,
            speed_based=speed_based,
            start=start,
            end=index,
        )
        if split is not None:
            splits.append(split)
        start = index
        boundary += chunk_m
        number += 1

    last = count - 1
    start_distance = _number(distances[start]) if start < len(distances) else None
    end_distance = _number(distances[last]) if last >= 0 else None
    if (
        start_distance is not None
        and end_distance is not None
        and end_distance - start_distance >= _MIN_TAIL_M
    ):
        tail = _segment(
            label=_chunk_label(number, chunk_km, final=True),
            times=times,
            distances=distances,
            altitudes=altitudes,
            heart_rates=heart_rates,
            watts=watts,
            speed_based=speed_based,
            start=start,
            end=last,
        )
        if tail is not None:
            splits.append(tail)
    return splits

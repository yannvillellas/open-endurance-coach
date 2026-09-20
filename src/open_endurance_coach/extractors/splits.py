"""Per-kilometre splits of an activity, computed from its streams."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from open_endurance_coach.schemas.intervals import ActivitySplit

MAX_SPLITS = 43
STREAM_TYPES = ("time", "distance", "altitude", "heartrate")
RIDE_STREAM_TYPES = (*STREAM_TYPES, "watts")
_KM = 1000.0
_WHOLE_KM_M = 0.5


def stream_types(speed_based: bool) -> tuple[str, ...]:
    """Streams to request: rides add watts, since power is their intensity metric."""
    return RIDE_STREAM_TYPES if speed_based else STREAM_TYPES


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _readings(distances: Sequence[Any], count: int) -> list[float | None]:
    readings: list[float | None] = []
    for index in range(count):
        number = _number(distances[index])
        readings.append(number if number is not None and number >= 0 else None)
    return readings


def _at(series: Sequence[Any], index: int) -> Any:
    return series[index] if index < len(series) else None


def _series(series: Sequence[Any], indexes: Sequence[int], *, signed: bool = False) -> list[float]:
    values = (_number(_at(series, index)) for index in indexes)
    return [value for value in values if value is not None and (signed or value >= 0)]


def _width_m(total: float, rows: int) -> float:
    """Smallest whole-kilometre width that keeps the row count within ``rows``."""
    return max(_KM, math.ceil(total / rows / _KM) * _KM)


def _label(row: int, width_km: int, total: float, *, last: bool, partial: bool = False) -> str:
    first = row * width_km + 1
    if not last:
        label = f"km {first}" if width_km == 1 else f"km {first}-{first + width_km - 1}"
        return label + (" (partial)" if partial else "")
    last_km = max(first, math.ceil(total / _KM))
    short = total - math.floor(total / _KM) * _KM > _WHOLE_KM_M
    marked = partial or short
    if last_km == first:
        return f"km {first} (partial)" if marked else f"km {first}"
    return f"km {first}-{last_km}" + (" (partial)" if marked else "")


def _segment(
    *,
    label: str,
    times: Sequence[Any],
    indexes: Sequence[int],
    start: int,
    end: int,
    meters: float,
    altitudes: Sequence[Any],
    heart_rates: Sequence[Any],
    watts: Sequence[Any],
    speed_based: bool,
) -> ActivitySplit | None:
    start_time = _number(times[start])
    end_time = _number(times[end])
    if start_time is None or end_time is None:
        return None
    elapsed = round(end_time - start_time)
    if elapsed <= 0 or round(meters, 1) <= 0:
        return None

    rates = _series(heart_rates, indexes)
    power = _series(watts, indexes)
    heights = _series(altitudes, indexes, signed=True)
    gain = sum(max(0.0, later - earlier) for earlier, later in pairwise(heights))
    loss = sum(max(0.0, earlier - later) for earlier, later in pairwise(heights))
    grade = (heights[-1] - heights[0]) / meters * 100 if len(heights) >= 2 else None

    return ActivitySplit(
        label=label,
        distance_m=round(meters, 1),
        time_s=elapsed,
        pace_s_per_km=None if speed_based else round(elapsed / (meters / _KM)),
        average_speed_kmh=(round((meters / _KM) / (elapsed / 3600), 1) if speed_based else None),
        average_watts=round(sum(power) / len(power)) if power and speed_based else None,
        average_heartrate=round(sum(rates) / len(rates)) if rates else None,
        max_heartrate=round(max(rates)) if rates else None,
        elevation_gain_m=round(gain, 1),
        elevation_loss_m=round(loss, 1),
        grade_pct=round(grade, 1) if grade is not None else None,
    )


def per_km_splits(
    streams: Mapping[str, Sequence[Any]],
    *,
    speed_based: bool = False,
    max_splits: int = MAX_SPLITS,
) -> list[ActivitySplit]:
    """Split an activity into consecutive kilometre rows.

    The row width is the smallest whole number of kilometres that keeps the row count
    within ``max_splits``: a marathon fits one row per kilometre at the default cap, and
    longer activities widen the rows (2 km, 3 km, ...). Rows are cut at exact kilometre
    boundaries and labelled with the range they cover; the final row is ``(partial)``
    when the stream stops mid-kilometre. The table ends at the stream's furthest usable sample (the
    last one for a monotonic stream), so a kilometre with no usable sample is left out rather than
    invented, and so is a final row with no measurable duration. Read-only: nothing is written to
    Intervals and no intervals need to exist on the activity. Returns [] without a usable time or
    distance series.
    """
    times = streams.get("time") or []
    distances = streams.get("distance") or []
    altitudes = streams.get("altitude") or []
    heart_rates = streams.get("heartrate") or []
    watts = streams.get("watts") or []
    if not times or not distances:
        return []

    count = min(len(times), len(distances))
    readings = _readings(distances, count)
    total = max((value for value in readings if value is not None), default=0.0)
    if total <= 0:
        return []

    rows = max_splits if isinstance(max_splits, int) and not isinstance(max_splits, bool) else 0
    rows = rows if rows >= 1 else MAX_SPLITS
    width_m = _width_m(total, rows)
    width_km = int(width_m // _KM)
    row_count = max(1, math.ceil(total / width_m))

    buckets: dict[int, list[int]] = {}
    for index, reading in enumerate(readings):
        if reading is None or reading > total:
            continue
        buckets.setdefault(min(int(reading // width_m), row_count - 1), []).append(index)

    splits: list[ActivitySplit] = []
    previous = False
    for row in range(row_count):
        indexes = buckets.get(row)
        if not indexes:
            previous = False
            continue
        following = buckets.get(row + 1)
        end_index = following[0] if following else indexes[-1]
        start_distance = row * width_m
        end_distance = min(start_distance + width_m, total)
        covered_start = start_distance if previous else readings[indexes[0]]
        covered_end = end_distance if following else readings[end_index]
        if covered_start is None or covered_end is None or covered_end <= covered_start:
            previous = True
            continue
        split = _segment(
            label=_label(
                row,
                width_km,
                total,
                last=row == row_count - 1,
                partial=covered_start != start_distance or covered_end != end_distance,
            ),
            times=times,
            indexes=indexes,
            start=indexes[0],
            end=end_index,
            meters=covered_end - covered_start,
            altitudes=altitudes,
            heart_rates=heart_rates,
            watts=watts,
            speed_based=speed_based,
        )
        if split is not None:
            splits.append(split)
        previous = True
    return splits

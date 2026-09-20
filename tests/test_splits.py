from typing import Any

from open_endurance_coach.extractors.splits import MAX_SPLITS, per_km_splits


def steady(metres: float, *, seconds_per_km: int = 300) -> dict[str, list]:
    speed = 1000 / seconds_per_km
    count = int(metres / speed)
    times = list(range(count + 1))
    distances = [round(index * speed, 4) for index in times]
    if distances[-1] < metres:
        times.append(times[-1] + 1)
        distances.append(metres)
    return {"time": times, "distance": distances}


def labels(splits: list) -> list[str]:
    return [split.label for split in splits]


def test_a_five_kilometre_run_is_one_row_per_kilometre() -> None:
    splits = per_km_splits(steady(5000))
    assert labels(splits) == ["km 1", "km 2", "km 3", "km 4", "km 5"]
    assert all(split.distance_m == 1000.0 for split in splits)
    assert all(split.pace_s_per_km == 300 for split in splits)


def test_a_marathon_fits_one_row_per_kilometre() -> None:
    splits = per_km_splits(steady(42195))
    assert len(splits) == 43
    assert labels(splits)[-1] == "km 43 (partial)"
    assert splits[-1].distance_m == 195.0


def test_exactly_forty_three_kilometres_is_not_partial() -> None:
    splits = per_km_splits(steady(43000))
    assert len(splits) == 43
    assert labels(splits)[-1] == "km 43"
    assert splits[-1].distance_m == 1000.0


def test_a_longer_activity_widens_the_rows() -> None:
    splits = per_km_splits(steady(43001))
    assert len(splits) == 22
    assert labels(splits)[0] == "km 1-2"
    assert labels(splits)[-1] == "km 43-44 (partial)"


def test_two_kilometre_rows_up_to_eighty_six_kilometres() -> None:
    splits = per_km_splits(steady(86000))
    assert len(splits) == 43
    assert labels(splits)[0] == "km 1-2"
    assert labels(splits)[-1] == "km 85-86"


def test_a_marathon_and_a_half_widens_to_three_kilometres() -> None:
    splits = per_km_splits(steady(86200))
    assert len(splits) == 29
    assert labels(splits)[0] == "km 1-3"
    assert labels(splits)[-1] == "km 85-87 (partial)"


def test_a_hundred_and_fifty_kilometres_uses_four_kilometre_rows() -> None:
    splits = per_km_splits(steady(150000))
    assert len(splits) == 38
    assert labels(splits)[0] == "km 1-4"
    assert labels(splits)[-1] == "km 149-150"
    assert splits[-1].distance_m == 2000.0
    assert "(partial)" not in splits[-1].label


def test_a_stream_that_stops_mid_kilometre_ends_partial() -> None:
    splits = per_km_splits(steady(9500))
    assert labels(splits)[-1] == "km 10 (partial)"
    assert splits[-1].distance_m == 500.0
    assert splits[-1].pace_s_per_km == 300
    assert sum(split.distance_m for split in splits) == 9500.0


def test_a_bool_cap_falls_back_to_the_default() -> None:
    assert len(per_km_splits(steady(5000), max_splits=True)) == 5


def test_just_over_a_kilometre_splits_in_two() -> None:
    splits = per_km_splits({"time": [0, 1, 2], "distance": [0.0, 1000.0, 1001.0]})
    assert labels(splits) == ["km 1", "km 2 (partial)"]
    assert [split.distance_m for split in splits] == [1000.0, 1.0]


def test_the_peak_ends_the_table_without_an_activity_distance() -> None:
    splits = per_km_splits(steady(2500))
    assert labels(splits) == ["km 1", "km 2", "km 3 (partial)"]
    assert splits[-1].distance_m == 500.0


def test_kilometres_without_samples_are_left_out() -> None:
    streams: dict[str, list[Any]] = {
        "time": [0, 500, 1000, 1500, 2000],
        "distance": [0.0, 500.0, 5100.0, 5150.0, 5200.0],
    }
    splits = per_km_splits(streams)
    assert labels(splits) == ["km 1", "km 6 (partial)"]
    assert [split.distance_m for split in splits] == [1000.0, 200.0]


def test_labels_cover_consecutive_kilometres_without_overlap() -> None:
    per_km = per_km_splits(steady(43000))
    assert labels(per_km) == [f"km {number}" for number in range(1, 44)]
    pairs = per_km_splits(steady(86000))
    assert labels(pairs) == [f"km {2 * row + 1}-{2 * row + 2}" for row in range(43)]


def test_rides_report_speed_and_power_and_runs_report_pace() -> None:
    streams = steady(2000)
    streams["watts"] = [200] * len(streams["time"])
    rides = per_km_splits(streams, speed_based=True)
    assert all(split.average_speed_kmh == 12.0 for split in rides)
    assert all(split.average_watts == 200 for split in rides)
    assert all(split.pace_s_per_km is None for split in rides)
    assert all(split.average_watts is None for split in per_km_splits(streams))


def test_climbing_and_descending_rows_report_gain_and_grade() -> None:
    climb: dict[str, list[Any]] = {
        "time": [0, 100, 200],
        "distance": [0.0, 500.0, 1000.0],
        "altitude": [100.0, 150.0, 200.0],
    }
    split = per_km_splits(climb)[0]
    assert split.elevation_gain_m == 100.0
    assert split.elevation_loss_m == 0.0
    assert split.grade_pct == 10.0

    descent: dict[str, list[Any]] = {
        "time": [0, 100, 200],
        "distance": [0.0, 500.0, 1000.0],
        "altitude": [200.0, 150.0, 100.0],
    }
    split = per_km_splits(descent)[0]
    assert split.elevation_loss_m == 100.0
    assert split.grade_pct == -10.0


def test_negative_heart_rate_and_watts_are_treated_as_missing() -> None:
    streams = steady(2000)
    streams["heartrate"] = [-40, 150] * 300 + [-40]
    streams["watts"] = [-100, 200] * 300 + [-100]
    rides = per_km_splits(streams, speed_based=True)
    assert all(split.average_heartrate == 150 for split in rides)
    assert all(split.average_watts == 200 for split in rides)


def test_all_negative_heart_rate_and_watts_yield_missing_values() -> None:
    streams = steady(2000)
    streams["heartrate"] = [-50] * len(streams["time"])
    streams["watts"] = [-50] * len(streams["time"])
    rows = per_km_splits(streams, speed_based=True)
    assert all(row.average_heartrate is None and row.max_heartrate is None for row in rows)
    assert all(row.average_watts is None for row in rows)


def test_zero_heart_rate_and_watts_are_real_readings() -> None:
    streams = steady(2000)
    streams["heartrate"] = [0] * len(streams["time"])
    streams["watts"] = [0] * len(streams["time"])
    rows = per_km_splits(streams, speed_based=True)
    assert all(row.average_heartrate == 0 and row.max_heartrate == 0 for row in rows)
    assert all(row.average_watts == 0 for row in rows)


def test_negative_altitude_stays_signed() -> None:
    streams: dict[str, list[Any]] = {
        "time": [0, 100, 200],
        "distance": [0.0, 500.0, 1000.0],
        "altitude": [-50.0, -100.0, -150.0],
    }
    split = per_km_splits(streams)[0]
    assert split.elevation_gain_m == 0.0
    assert split.elevation_loss_m == 100.0
    assert split.grade_pct == -10.0


def test_hill_heart_rate_and_power_are_averaged_per_row() -> None:
    streams = steady(2000)
    streams["heartrate"] = [150, 170] * 1500 + [150]
    climbs = per_km_splits(streams)
    assert climbs[0].average_heartrate == 160
    assert climbs[0].max_heartrate == 170


def test_unusable_samples_are_skipped() -> None:
    streams: dict[str, list[Any]] = {
        "time": [0, 1, 2, 3, 4, 5],
        "distance": [0.0, None, 1000.0, float("nan"), 2000.0, -5.0],
    }
    assert labels(per_km_splits(streams)) == ["km 1", "km 2"]


def test_missing_series_yield_no_splits() -> None:
    assert per_km_splits({}) == []
    assert per_km_splits({"time": [0, 1], "distance": [None, None]}) == []
    assert per_km_splits({"time": [], "distance": [0.0]}) == []


def test_a_thousand_kilometres_stays_within_the_cap() -> None:
    splits = per_km_splits(steady(1_000_000))
    assert 0 < len(splits) <= MAX_SPLITS
    assert sum(split.distance_m for split in splits) == 1_000_000.0

from open_endurance_coach.extractors.splits import per_km_splits


def steady_streams(seconds: int, *, seconds_per_km: int = 300) -> dict[str, list]:
    speed = 1000 / seconds_per_km
    times = list(range(seconds + 1))
    return {"time": times, "distance": [round(index * speed, 4) for index in times]}


def test_per_km_splits_of_a_steady_run() -> None:
    streams = steady_streams(900)
    streams["heartrate"] = [150] * 901
    streams["altitude"] = [100.0] * 901
    splits = per_km_splits(streams)

    assert [split.label for split in splits] == ["km 1", "km 2", "km 3"]
    assert all(split.pace_s_per_km == 300 for split in splits)
    assert all(split.average_heartrate == 150 for split in splits)
    assert all(split.elevation_gain_m == 0 for split in splits)


def test_per_km_splits_include_a_partial_tail() -> None:
    splits = per_km_splits(steady_streams(1050))
    assert [split.label for split in splits] == ["km 1", "km 2", "km 3", "km 4 (partial)"]
    assert splits[-1].distance_m == 500.0
    assert splits[-1].time_s == 150


def test_per_km_splits_report_climb_and_descent() -> None:
    streams = steady_streams(900)
    altitudes = []
    for index in range(901):
        if index <= 300:
            altitudes.append(100.0 + index * 50 / 300)
        elif index <= 600:
            altitudes.append(150.0 + (index - 300) * 50 / 300)
        else:
            altitudes.append(200.0 - (index - 600) * 100 / 300)
    streams["altitude"] = altitudes

    splits = per_km_splits(streams)
    climb = splits[0]
    descent = splits[2]
    assert climb.elevation_gain_m == 50.0
    assert climb.grade_pct == 5.0
    assert descent.elevation_loss_m == 100.0
    assert descent.grade_pct == -10.0


def test_per_km_splits_tolerate_missing_series() -> None:
    splits = per_km_splits(steady_streams(600))
    assert len(splits) == 2
    assert all(split.average_heartrate is None for split in splits)
    assert all(split.max_heartrate is None for split in splits)
    assert all(split.grade_pct is None for split in splits)
    assert all(split.elevation_gain_m == 0 for split in splits)


def test_per_km_splits_need_time_and_distance() -> None:
    assert per_km_splits({}) == []
    assert per_km_splits({"time": [0, 1, 2]}) == []
    assert per_km_splits({"distance": [0.0, 500.0, 1200.0]}) == []

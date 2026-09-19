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


def test_speed_based_splits_report_speed_and_power() -> None:
    streams = steady_streams(600)
    streams["watts"] = [200] * 601
    splits = per_km_splits(streams, speed_based=True)

    assert [split.label for split in splits] == ["km 1", "km 2"]
    assert all(split.average_speed_kmh == 12.0 for split in splits)
    assert all(split.average_watts == 200 for split in splits)
    assert all(split.pace_s_per_km is None for split in splits)


def test_long_activity_splits_are_capped_by_merging_kilometres() -> None:
    splits = per_km_splits(steady_streams(9000), max_splits=10)
    assert len(splits) == 10
    assert splits[0].label == "km 1-3"
    assert splits[-1].label == "km 28-30"
    assert all(split.distance_m == 3000.0 for split in splits)


def test_default_cap_keeps_per_kilometre_up_to_forty() -> None:
    splits = per_km_splits(steady_streams(9900))
    assert len(splits) == 33
    assert splits[0].label == "km 1"
    assert splits[-1].label == "km 33"


def test_default_cap_merges_beyond_forty_kilometres() -> None:
    splits = per_km_splits(steady_streams(15000))
    assert len(splits) == 25
    assert splits[0].label == "km 1-2"
    assert splits[-1].label == "km 49-50"


def test_forty_kilometres_stay_per_kilometre_at_the_default_cap() -> None:
    splits = per_km_splits(steady_streams(12000))
    assert len(splits) == 40
    assert splits[0].label == "km 1"
    assert splits[-1].label == "km 40"
    assert all(split.distance_m == 1000.0 for split in splits)


def test_nonpositive_max_splits_falls_back_to_the_default_cap() -> None:
    for max_splits in (0, -1):
        splits = per_km_splits(steady_streams(15000), max_splits=max_splits)
        assert len(splits) == 25
        assert splits[0].label == "km 1-2"
        assert splits[-1].label == "km 49-50"


def test_corrupted_final_distance_sample_keeps_earlier_splits() -> None:
    streams = steady_streams(900)
    streams["distance"][-1] = 0.0
    splits = per_km_splits(streams)
    assert [split.label for split in splits] == ["km 1", "km 2"]


def test_per_km_splits_need_time_and_distance() -> None:
    assert per_km_splits({}) == []
    assert per_km_splits({"time": [0, 1, 2]}) == []
    assert per_km_splits({"distance": [0.0, 500.0, 1200.0]}) == []

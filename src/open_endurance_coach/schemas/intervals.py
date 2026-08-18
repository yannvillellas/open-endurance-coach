from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class Interval(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    type: str | None = None
    label: str | None = None
    zone: int | None = None
    intensity: float | None = None
    average_watts: float | None = None
    average_heartrate: float | None = None
    max_watts: float | None = None
    max_heartrate: float | None = None
    moving_time: int | None = None
    training_load: float | None = None
    start_time: int | None = None
    end_time: int | None = None
    gap: float | None = None


class Activity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    start_date_local: datetime
    type: str
    name: str
    icu_training_load: float | None = None
    moving_time: int | None = None
    distance: float | None = None
    total_elevation_gain: float | None = None
    icu_average_watts: float | None = None
    icu_weighted_avg_watts: float | None = None
    average_heartrate: float | None = None
    max_heartrate: float | None = None
    average_cadence: float | None = None
    icu_rpe: float | None = None
    perceived_exertion: float | None = None
    session_rpe: float | None = None
    icu_intensity: float | None = None
    icu_ctl: float | None = None
    icu_atl: float | None = None
    icu_intervals: list[Interval] | None = None


class SportInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    eftp: float | None = None
    wPrime: float | None = None
    pMax: float | None = None


class Wellness(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: date
    ctl: float | None = None
    atl: float | None = None
    rampRate: float | None = None
    hrv: float | None = None
    hrvSDNN: float | None = None
    restingHR: int | None = None
    sleepSecs: int | None = None
    sleepQuality: float | None = None
    fatigue: float | None = None
    stress: float | None = None
    soreness: float | None = None
    readiness: float | None = None
    weight: float | None = None
    vo2max: float | None = None
    updated: datetime | None = None
    sportInfo: list[SportInfo] | None = None


class SportSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    athlete_id: str | int | None = None
    types: list[str] | None = None
    ftp: float | None = None
    indoor_ftp: float | None = None
    lthr: int | None = None
    max_hr: int | None = None
    w_prime: float | None = None
    p_max: float | None = None
    power_zones: list[float] | None = None
    hr_zones: list[int] | None = None


class Event(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | str | None = None
    name: str
    start_date_local: datetime
    category: str | None = None
    description: str | None = None
    end_date_local: datetime | None = None
    type: str | None = None
    moving_time: int | None = None
    icu_training_load: float | None = None
    workout_doc: dict[str, Any] | None = None
    plan_folder_id: int | str | None = None
    plan_workout_id: int | str | None = None

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from open_endurance_coach.schemas import intervals

SPEC_PATH = Path(__file__).parent / "spec" / "openapi.json"

MODEL_SPEC_SCHEMAS: list[tuple[type[BaseModel], str]] = [
    (intervals.Activity, "ActivityWithIntervals"),
    (intervals.Interval, "Interval"),
    (intervals.Wellness, "Wellness"),
    (intervals.SportInfo, "SportInfo"),
    (intervals.SportSettings, "SportSettings"),
    (intervals.Event, "Event"),
]


def load_spec() -> dict[str, Any]:
    with SPEC_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.mark.parametrize("model, schema_name", MODEL_SPEC_SCHEMAS)
def test_model_fields_exist_in_spec(model: type[BaseModel], schema_name: str) -> None:
    schemas = load_spec()["components"]["schemas"]
    properties = schemas[schema_name]["properties"]
    missing = set(model.model_fields) - set(properties)
    assert not missing, (
        f"{model.__name__} fields missing from spec schema {schema_name}: {sorted(missing)}"
    )

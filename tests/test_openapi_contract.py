import json
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).parent / "spec" / "openapi.json"

REQUIRED_ENDPOINTS: list[tuple[str, str]] = [
    ("get", "/api/v1/athlete/{id}/activities"),
    ("get", "/api/v1/activity/{id}"),
    ("get", "/api/v1/athlete/{id}/wellness"),
    ("get", "/api/v1/athlete/{id}/events"),
    ("post", "/api/v1/athlete/{id}/events"),
    ("put", "/api/v1/athlete/{id}/events/{eventId}"),
    ("delete", "/api/v1/athlete/{id}/events/{eventId}"),
    ("get", "/api/v1/athlete/{id}/sport-settings"),
    ("get", "/api/v1/athlete/{id}/athlete-summary"),
    ("get", "/api/v1/athlete/{id}/activities/interval-search"),
]

REQUIRED_PARAMETERS: dict[str, set[str]] = {
    "/api/v1/athlete/{id}/activities": {"oldest", "newest", "fields"},
    "/api/v1/activity/{id}": {"intervals"},
    "/api/v1/athlete/{id}/wellness": {"oldest", "newest", "cols"},
    "/api/v1/athlete/{id}/events": {"oldest", "newest", "category"},
}


def load_spec() -> dict[str, Any]:
    assert SPEC_PATH.exists(), "OpenAPI snapshot missing: run refresh-spec"
    with SPEC_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalized_paths(spec: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for path, methods in spec.get("paths", {}).items():
        stripped = path.replace("{ext}", "").replace("{format}", "").replace("{athleteId}", "{id}")
        normalized.setdefault(stripped, {}).update(methods)
    return normalized


def test_snapshot_is_a_valid_openapi_spec() -> None:
    spec = load_spec()
    assert spec.get("openapi", "").startswith("3.")
    assert spec.get("info", {}).get("title") == "Intervals.icu API"
    assert spec.get("paths"), "spec contains no paths"


def test_consumed_endpoints_exist_in_spec() -> None:
    paths = normalized_paths(load_spec())
    for method, path in REQUIRED_ENDPOINTS:
        assert path in paths, f"endpoint missing from spec: {method.upper()} {path}"
        assert method in paths[path], f"method missing from spec: {method.upper()} {path}"


def test_consumed_parameters_exist_in_spec() -> None:
    paths = normalized_paths(load_spec())
    for path, expected in REQUIRED_PARAMETERS.items():
        operation = paths[path].get("get", {})
        available = {param.get("name") for param in operation.get("parameters", [])}
        missing = expected - available
        assert not missing, f"GET {path}: parameters missing from spec: {sorted(missing)}"


def test_api_key_basic_auth_scheme_is_documented() -> None:
    spec = load_spec()
    schemes = spec.get("components", {}).get("securitySchemes", {})
    api_key = schemes.get("APIKey")
    assert api_key, "APIKey security scheme missing from spec"
    assert api_key.get("type") == "http"
    assert api_key.get("scheme") == "basic"

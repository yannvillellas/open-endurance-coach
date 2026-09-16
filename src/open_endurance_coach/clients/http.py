from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx


def error_detail(response: httpx.Response) -> str:
    """A short, sanitized reason from an error response - never the raw body."""
    try:
        data = response.json()
    except ValueError:
        return ""
    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:120]
    return ""


def parse_retry_after(
    headers: httpx.Headers, default: float, *, now: datetime | None = None
) -> float:
    value = headers.get("Retry-After")
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return default
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    return max(0.0, (target - (now or datetime.now(UTC))).total_seconds())

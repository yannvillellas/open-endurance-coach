import math
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

MAX_RETRY_AFTER_SECONDS = 300.0


def error_detail(response: httpx.Response) -> str:
    """A short, single-line reason from an error response - never the raw body."""
    try:
        data = response.json()
    except ValueError:
        return ""
    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())[:120]
    return ""


def parse_retry_after(
    headers: httpx.Headers, default: float, *, now: datetime | None = None
) -> float:
    value = headers.get("Retry-After")
    if value is None:
        return default
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        if not math.isfinite(seconds):
            return default
        return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)
    try:
        target = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return default
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    delay = (target - (now or datetime.now(UTC))).total_seconds()
    return min(max(delay, 0.0), MAX_RETRY_AFTER_SECONDS)

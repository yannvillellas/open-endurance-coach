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


def parse_retry_after(headers: httpx.Headers, default: float) -> float:
    value = headers.get("Retry-After")
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

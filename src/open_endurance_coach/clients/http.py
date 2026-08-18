import httpx


def parse_retry_after(headers: httpx.Headers, default: float) -> float:
    value = headers.get("Retry-After")
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

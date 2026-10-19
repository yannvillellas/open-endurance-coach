import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from open_endurance_coach.config import Settings

from .http import error_detail, parse_retry_after


def _path_segment(value: str) -> str:
    """Quote an id for a URL path so it cannot escape its segment."""
    return quote(str(value), safe="")


BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


RATE_LIMIT_SLOW_RATIO = 0.2
RATE_LIMIT_CRAWL_RATIO = 0.1
RATE_LIMIT_COOLDOWN_SECONDS = 60.0


@dataclass
class RateLimits:
    limit_15m: int | None = None
    remaining_15m: int | None = None
    limit_daily: int | None = None
    remaining_daily: int | None = None

    @classmethod
    def from_headers(cls, headers: httpx.Headers) -> "RateLimits":
        limits = headers.get("X-RateLimit-Limit", "")
        remaining = headers.get("X-RateLimit-Remaining", "")

        def pair(value: str) -> tuple[int | None, int | None]:
            parts = value.split(",")
            if len(parts) != 2:
                return None, None
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                return None, None

        limit_15m, limit_daily = pair(limits)
        remaining_15m, remaining_daily = pair(remaining)
        return cls(
            limit_15m=limit_15m,
            remaining_15m=remaining_15m,
            limit_daily=limit_daily,
            remaining_daily=remaining_daily,
        )

    def pressure(self) -> float | None:
        """The tightest remaining/limit ratio across the windows, None when unknown."""
        ratios: list[float] = []
        if self.limit_15m and self.remaining_15m is not None:
            ratios.append(self.remaining_15m / self.limit_15m)
        if self.limit_daily and self.remaining_daily is not None:
            ratios.append(self.remaining_daily / self.limit_daily)
        return min(ratios) if ratios else None


class IntervalsApiError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        message: str,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(f"Intervals.icu API error {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.retry_after = retry_after


class IntervalsClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._settings = settings
        self._sleep = sleep or asyncio.sleep
        self._client = httpx.AsyncClient(
            base_url=settings.intervals_base_url,
            auth=httpx.BasicAuth("API_KEY", settings.intervals_api_key),
            headers={
                "User-Agent": BROWSER_USER_AGENT,
                "Accept": "application/json",
            },
            transport=transport,
            timeout=60.0,
            follow_redirects=True,
        )
        self._min_interval = 1.0 / settings.requests_per_second
        self._last_request_at = 0.0
        self._cooldown_until = 0.0
        self._throttle_lock = asyncio.Lock()
        self.rate_limits = RateLimits()

    def _athlete_path(self, path: str) -> str:
        return f"/athlete/{self._settings.intervals_athlete_id}{path}"

    def _pacing_interval(self) -> float:
        """Slow down before the quota runs out, from the last observed headers."""
        pressure = self.rate_limits.pressure()
        if pressure is None:
            return self._min_interval
        if pressure <= RATE_LIMIT_CRAWL_RATIO:
            return self._min_interval * 4
        if pressure <= RATE_LIMIT_SLOW_RATIO:
            return self._min_interval * 2
        return self._min_interval

    def _exhausted(self) -> bool:
        pressure = self.rate_limits.pressure()
        return pressure is not None and pressure <= 0

    async def _throttle(self) -> None:
        async with self._throttle_lock:
            now = time.monotonic()
            if self._exhausted():
                # One shared cooldown: concurrent callers wait it once, then pace.
                self._cooldown_until = max(self._cooldown_until, now + RATE_LIMIT_COOLDOWN_SECONDS)
            ready_at = max(self._last_request_at + self._pacing_interval(), self._cooldown_until)
            wait = max(0.0, ready_at - now)
            self._last_request_at = now + wait
        if wait > 0:
            await self._sleep(wait)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        retry_ambiguous: bool = True,
    ) -> httpx.Response:
        await self._throttle()
        response: httpx.Response | None = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                response = await self._client.request(method, path, params=params, json=payload)
            except httpx.TransportError:
                response = None
            if response is not None:
                self.rate_limits = RateLimits.from_headers(response.headers)
                if response.status_code == 429 and attempt < self._settings.max_retries:
                    retry_after = parse_retry_after(response.headers, default=60.0)
                    await self._sleep(retry_after)
                    continue
                if (
                    response.status_code >= 500
                    and attempt < self._settings.max_retries
                    and retry_ambiguous
                ):
                    await self._sleep(self._settings.retry_base_delay * 2**attempt)
                    continue
                break
            if not retry_ambiguous:
                break
            if attempt < self._settings.max_retries:
                await self._sleep(self._settings.retry_base_delay * 2**attempt)
        if response is None:
            raise IntervalsApiError(0, "transport error after retries")
        if response.status_code >= 400:
            error_retry_after: float | None = None
            if response.status_code == 429:
                error_retry_after = parse_retry_after(response.headers, default=0.0) or None
            raise IntervalsApiError(
                response.status_code,
                error_detail(response) or "request rejected",
                retry_after=error_retry_after,
            )
        return response

    def _json(self, response: httpx.Response, label: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise IntervalsApiError(response.status_code, f"unexpected {label} payload") from exc

    def _json_list(self, response: httpx.Response, label: str) -> list[Any]:
        data = self._json(response, label)
        if not isinstance(data, list):
            raise IntervalsApiError(response.status_code, f"unexpected {label} payload")
        return data

    def _json_object(self, response: httpx.Response, label: str) -> dict[str, Any]:
        data = self._json(response, label)
        if not isinstance(data, dict):
            raise IntervalsApiError(response.status_code, f"unexpected {label} payload")
        return data

    async def list_activities(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        params = {"oldest": oldest, "newest": newest}
        response = await self._request("GET", self._athlete_path("/activities"), params=params)
        return self._json_list(response, "activities")

    async def get_activity(self, activity_id: str, intervals: bool = True) -> dict[str, Any]:
        params = {"intervals": str(intervals).lower()}
        response = await self._request(
            "GET", f"/activity/{_path_segment(activity_id)}", params=params
        )
        return self._json_object(response, "activity")

    async def get_activity_streams(
        self, activity_id: str, types: Sequence[str]
    ) -> dict[str, list[Any]]:
        response = await self._request(
            "GET",
            f"/activity/{_path_segment(activity_id)}/streams",
            params={"types": ",".join(types)},
        )
        streams: dict[str, list[Any]] = {}
        for row in self._json_list(response, "streams"):
            if not isinstance(row, dict):
                continue
            name = row.get("type")
            data = row.get("data")
            if isinstance(name, str) and isinstance(data, list):
                streams[name] = data
        return streams

    async def list_wellness(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        response = await self._request(
            "GET", self._athlete_path("/wellness"), params={"oldest": oldest, "newest": newest}
        )
        return self._json_list(response, "wellness")

    async def list_events(
        self,
        oldest: str,
        newest: str,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"oldest": oldest, "newest": newest}
        if category:
            params["category"] = category
        response = await self._request("GET", self._athlete_path("/events"), params=params)
        return self._json_list(response, "events")

    async def get_event(self, event_id: str) -> dict[str, Any]:
        response = await self._request(
            "GET", self._athlete_path(f"/events/{_path_segment(event_id)}")
        )
        return self._json_object(response, "event")

    async def create_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        # A create is not idempotent: a retried POST can duplicate the event when the
        # first one persisted but its response was lost. The writer reconciles instead.
        response = await self._request(
            "POST", self._athlete_path("/events"), payload=payload, retry_ambiguous=False
        )
        return self._json_object(response, "event")

    async def update_event(self, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request(
            "PUT", self._athlete_path(f"/events/{_path_segment(event_id)}"), payload=payload
        )
        return self._json_object(response, "event")

    async def delete_event(self, event_id: str) -> None:
        await self._request("DELETE", self._athlete_path(f"/events/{_path_segment(event_id)}"))

    async def get_sport_settings(self) -> list[dict[str, Any]]:
        response = await self._request("GET", self._athlete_path("/sport-settings"))
        return self._json_list(response, "sport-settings")

    async def get_athlete_summary(
        self, *, start: str | None = None, end: str | None = None
    ) -> list[dict[str, Any]]:
        params = {key: value for key, value in (("start", start), ("end", end)) if value}
        response = await self._request("GET", self._athlete_path("/athlete-summary"), params=params)
        return self._json_list(response, "athlete-summary")

    async def aclose(self) -> None:
        await self._client.aclose()

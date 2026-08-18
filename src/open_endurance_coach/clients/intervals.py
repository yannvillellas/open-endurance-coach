import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from open_endurance_coach.config import Settings

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


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
        self.rate_limits = RateLimits()

    def _athlete_path(self, path: str) -> str:
        return f"/athlete/{self._settings.intervals_athlete_id}{path}"

    async def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            await self._sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
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
                    retry_after = float(response.headers.get("Retry-After", 60))
                    await self._sleep(retry_after)
                    continue
                if response.status_code >= 500 and attempt < self._settings.max_retries:
                    await self._sleep(self._settings.retry_base_delay * 2**attempt)
                    continue
                break
            if attempt < self._settings.max_retries:
                await self._sleep(self._settings.retry_base_delay * 2**attempt)
        if response is None:
            raise IntervalsApiError(0, "transport error after retries")
        if response.status_code >= 400:
            error_retry_after: float | None = None
            if response.status_code == 429:
                error_retry_after = float(response.headers.get("Retry-After", 0)) or None
            raise IntervalsApiError(
                response.status_code,
                response.text[:500],
                retry_after=error_retry_after,
            )
        return response

    async def list_activities(
        self,
        oldest: str,
        newest: str,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"oldest": oldest, "newest": newest}
        if fields:
            params["fields"] = ",".join(fields)
        response = await self._request("GET", self._athlete_path("/activities"), params=params)
        return response.json()

    async def get_activity(self, activity_id: str, intervals: bool = True) -> dict[str, Any]:
        params = {"intervals": str(intervals).lower()}
        response = await self._request("GET", f"/activity/{activity_id}", params=params)
        return response.json()

    async def list_wellness(self, oldest: str, newest: str) -> list[dict[str, Any]]:
        response = await self._request(
            "GET", self._athlete_path("/wellness"), params={"oldest": oldest, "newest": newest}
        )
        return response.json()

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
        return response.json()

    async def get_event(self, event_id: str) -> dict[str, Any]:
        response = await self._request("GET", self._athlete_path(f"/events/{event_id}"))
        return response.json()

    async def create_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request("POST", self._athlete_path("/events"), payload=payload)
        return response.json()

    async def update_event(self, event_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request(
            "PUT", self._athlete_path(f"/events/{event_id}"), payload=payload
        )
        return response.json()

    async def delete_event(self, event_id: str) -> None:
        await self._request("DELETE", self._athlete_path(f"/events/{event_id}"))

    async def get_sport_settings(self) -> list[dict[str, Any]]:
        response = await self._request("GET", self._athlete_path("/sport-settings"))
        return response.json()

    async def get_athlete(self) -> dict[str, Any]:
        response = await self._request("GET", self._athlete_path(""))
        return response.json()

    async def get_athlete_summary(self) -> list[dict[str, Any]]:
        response = await self._request("GET", self._athlete_path("/athlete-summary"))
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()

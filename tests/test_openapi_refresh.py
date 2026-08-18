import httpx
import pytest

from open_endurance_coach.spec.refresh import SPEC_URL, download_spec


async def test_download_spec_fetches_docs_url_and_parses_json() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == SPEC_URL
        return httpx.Response(200, json={"openapi": "3.0.1", "paths": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        spec = await download_spec(client)
    finally:
        await client.aclose()
    assert spec == {"openapi": "3.0.1", "paths": {}}


async def test_download_spec_rejects_non_spec_payloads() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hello": "world"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError, match="OpenAPI spec"):
            await download_spec(client)
    finally:
        await client.aclose()


async def test_download_spec_raises_on_http_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await download_spec(client)
    finally:
        await client.aclose()

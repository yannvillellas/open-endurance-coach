import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from open_endurance_coach.clients.intervals import BROWSER_USER_AGENT

SPEC_URL = "https://intervals.icu/api/v1/docs"


def _default_output_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "tests" / "spec" / "openapi.json"


async def download_spec(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get(SPEC_URL)
    response.raise_for_status()
    spec = response.json()
    if not isinstance(spec, dict) or not spec.get("openapi"):
        raise RuntimeError(f"{SPEC_URL} did not return an OpenAPI spec")
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the vendored Intervals.icu OpenAPI snapshot"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_default_output_path(),
        help="output path (default: tests/spec/openapi.json)",
    )
    args = parser.parse_args()
    output_path: Path = args.out

    async def run() -> None:
        async with httpx.AsyncClient(
            headers={"User-Agent": BROWSER_USER_AGENT, "Accept": "application/json"},
            timeout=60.0,
            follow_redirects=True,
        ) as client:
            spec = await download_spec(client)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"wrote {output_path}")

    asyncio.run(run())


if __name__ == "__main__":
    main()

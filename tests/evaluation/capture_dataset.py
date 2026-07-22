import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from news.digest.llm_client import LlmClient
from news.digest.miniflux_client import MinifluxClient
from news.digest.service import DigestService
from news.settings import Settings

DATA_DIR = Path(__file__).parent / "data"


async def capture(
    settings: Settings, *, category: str, now: datetime | None = None
) -> list[dict[str, Any]]:
    client = MinifluxClient(
        str(settings.miniflux_api_base),
        settings.miniflux_api_key,
        httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)),
    )
    service = DigestService(
        settings,
        client,
        cast(LlmClient, object()),
    )
    try:
        entries = await service.fetch_entries(
            category=category,
            now=now or datetime.now(UTC),
        )
    finally:
        await client.aclose()
    return [entry.model_dump(mode="json") for entry in entries]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture RSS entries for evaluation dataset")
    parser.add_argument(
        "--category",
        choices=["news", "Economy", "Technology"],
        default="news",
        help="RSS category to fetch (default: news)",
    )
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC),
        default=None,
        help="Date in YYYY-MM-DD format (default: today)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings()  # type: ignore

    now = args.date or datetime.now(UTC)
    category: str = args.category

    data = asyncio.run(
        capture(
            settings,
            category=category,
            now=now,
        )
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    date_str = now.strftime("%Y_%m_%d")
    filename = f"rss_entries_{date_str}_{category}.json"
    path = DATA_DIR / filename
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {len(data)} entries to {path}")


if __name__ == "__main__":
    main()

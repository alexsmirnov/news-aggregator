import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from news import config
from news.digest.miniflux_client import MinifluxClient
from news.digest.service import fetch_entries
from news.settings import Settings

DATA_DIR = Path(__file__).parent / "data"


async def capture(
    settings: Settings, *, category: str, now: datetime | None = None
) -> list[dict[str, Any]]:
    client = MinifluxClient(
        str(settings.miniflux_api_base), settings.miniflux_api_key
    )
    try:
        entries = await fetch_entries(
            client,
            category=category,
            lookback_hours=settings.fetch_lookback_hours,
            limit=settings.fetch_limit,
            max_chars=settings.entry_content_max_chars,
            now=now or datetime.now(UTC),
        )
    finally:
        await client.aclose()
    return [entry.model_dump(mode="json") for entry in entries]


def main() -> None:
    settings = Settings()
    data = asyncio.run(
        capture(
            settings,
            category=config.AGGREGATIONS[0].miniflux_category,
        )
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "rss_entries.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {len(data)} entries to {path}")


if __name__ == "__main__":
    main()

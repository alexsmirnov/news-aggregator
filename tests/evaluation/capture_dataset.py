import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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
        str(settings.miniflux_api_base), settings.miniflux_api_key
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


def main() -> None:
    settings = Settings()
    data = asyncio.run(
        capture(
            settings,
            category=settings.aggregations[0].miniflux_category,
        )
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "rss_entries.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {len(data)} entries to {path}")


if __name__ == "__main__":
    main()

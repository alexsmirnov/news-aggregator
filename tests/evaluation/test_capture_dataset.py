import json
from datetime import datetime
from unittest.mock import AsyncMock

import capture_dataset

from news.digest.schemas import RssEntry
from news.settings import Settings

NOW = datetime(2026, 7, 17, 12, 0, 0)


async def test_capture_serializes_entries(monkeypatch) -> None:
    # Arrange
    settings = Settings(
        miniflux_api_base="http://m.test",
        miniflux_api_key="k",
        litellm_api_key="lk",
        litellm_router="http://router.test",
        digest_output_dir="/tmp/digests",
        fetch_lookback_hours=24,
        fetch_limit=10000,
        entry_content_max_chars=1000,
    )
    entries = [
        RssEntry(
            id=1,
            title="T",
            link="http://a",
            content="C",
            published_at="2026-07-16",
            source="F",
        )
    ]
    fetch_entries = AsyncMock(return_value=entries)
    monkeypatch.setattr(capture_dataset, "fetch_entries", fetch_entries)

    # Act
    data = await capture_dataset.capture(settings, category="news", now=NOW)

    # Assert
    assert data == [
        {
            "id": 1,
            "title": "T",
            "link": "http://a",
            "content": "C",
            "published_at": "2026-07-16",
            "source": "F",
        }
    ]
    fetch_entries.assert_awaited_once()
    assert fetch_entries.call_args.kwargs["category"] == "news"
    assert fetch_entries.call_args.kwargs["lookback_hours"] == 24
    assert fetch_entries.call_args.kwargs["limit"] == 10000
    assert fetch_entries.call_args.kwargs["max_chars"] == 1000


def test_main_writes_snapshot_file(monkeypatch, tmp_path) -> None:
    # Arrange
    settings = Settings(
        miniflux_api_base="http://m.test",
        miniflux_api_key="k",
        litellm_api_key="lk",
        litellm_router="http://router.test",
        digest_output_dir="/tmp/digests",
        fetch_lookback_hours=24,
        fetch_limit=10000,
        entry_content_max_chars=1000,
    )
    capture = AsyncMock(return_value=[{"id": 1}])
    monkeypatch.setattr(capture_dataset, "Settings", lambda: settings)
    monkeypatch.setattr(capture_dataset, "capture", capture)
    monkeypatch.setattr(capture_dataset, "DATA_DIR", tmp_path)

    # Act
    capture_dataset.main()

    # Assert
    snapshot = tmp_path / "rss_entries.json"
    assert json.loads(snapshot.read_text()) == [{"id": 1}]
    capture.assert_awaited_once_with(settings, category="news")

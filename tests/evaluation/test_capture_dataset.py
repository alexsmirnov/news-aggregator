import json
import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

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
    fetch_entries_mock = AsyncMock(return_value=entries)
    service_instance = MagicMock()
    service_instance.fetch_entries = fetch_entries_mock
    digest_service_class = MagicMock(return_value=service_instance)
    monkeypatch.setattr(capture_dataset, "DigestService", digest_service_class)

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
    digest_service_class.assert_called_once()
    assert digest_service_class.call_args.args[0] is settings
    fetch_entries_mock.assert_awaited_once()
    assert fetch_entries_mock.call_args.kwargs == {
        "category": "news",
        "now": NOW,
    }


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

    monkeypatch.setattr(sys, "argv", ["capture_dataset"])

    # Act
    capture_dataset.main()

    # Assert
    today_str = datetime.now(UTC).strftime("%Y_%m_%d")
    snapshot = tmp_path / f"rss_entries_{today_str}_news.json"
    assert json.loads(snapshot.read_text()) == [{"id": 1}]
    capture.assert_awaited_once()

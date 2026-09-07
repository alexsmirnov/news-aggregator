from collections.abc import Callable
from unittest.mock import AsyncMock

import httpx
import pydantic
import pytest
from pydantic import SecretStr

from news.digest.miniflux_client import MinifluxClient, miniflux_client
from news.digest.schemas import RssEntry
from news.settings import Settings


def _sequence_handler(
    effects: list[httpx.Response | Exception],
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        effect = effects[len(calls) - 1]
        if isinstance(effect, Exception):
            raise effect
        return effect

    return handler, calls


def _constant_handler(
    effect: httpx.Response | Exception,
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if isinstance(effect, Exception):
            raise effect
        return effect

    return handler, calls


@pytest.fixture
def make_client() -> Callable[..., MinifluxClient]:
    def _make(
        handler: Callable[[httpx.Request], httpx.Response], **kwargs: int
    ) -> MinifluxClient:
        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        return MinifluxClient(
            base_url="http://miniflux.test",
            api_key=SecretStr("secret"),
            client=client,
            max_wait=0,
            **kwargs,
        )

    return _make


async def test_get_category_id_returns_matching_id(make_client):
    # Arrange
    handler, calls = _constant_handler(
        httpx.Response(
            200,
            json=[{"id": 3, "title": "tech"}, {"id": 7, "title": "news"}],
        )
    )
    client = make_client(handler)

    # Act
    category_id = await client.get_category_id("news")

    # Assert
    assert category_id == 7
    assert len(calls) == 1
    assert calls[0].method == "GET"
    assert str(calls[0].url) == "http://miniflux.test/v1/categories"
    assert calls[0].headers["X-Auth-Token"] == "secret"


async def test_get_category_id_unknown_title_raises(make_client):
    # Arrange
    handler, _ = _constant_handler(
        httpx.Response(200, json=[{"id": 3, "title": "tech"}])
    )
    client = make_client(handler)

    # Act / Assert
    with pytest.raises(LookupError):
        await client.get_category_id("news")


async def test_get_entries_resolves_category_and_returns_entries(make_client):
    # Arrange
    handler, calls = _sequence_handler(
        [
            httpx.Response(200, json=[{"id": 7, "title": "news"}]),
            httpx.Response(
                200,
                json={
                    "total": 1,
                    "entries": [
                        {
                            "id": 1,
                            "title": "T",
                            "url": "http://x",
                            "content": "<p>c</p>",
                            "published_at": "2026-07-16",
                            "feed": {"title": "F"},
                        }
                    ],
                },
            ),
        ]
    )
    client = make_client(handler)

    # Act
    entries = await client.get_entries(
        "news",
        published_after=1752700000,
        published_before=1752800000,
        order="published_at",
        limit=5,
    )

    # Assert
    assert len(calls) == 2
    assert calls[0].url.path == "/v1/categories"
    assert calls[1].url.path == "/v1/categories/7/entries"
    assert calls[1].url.params["published_after"] == "1752700000"
    assert calls[1].url.params["published_before"] == "1752800000"
    assert calls[1].url.params["order"] == "published_at"
    assert calls[1].url.params["limit"] == "5"
    assert calls[1].url.params["offset"] == "0"
    assert len(entries) == 1
    assert isinstance(entries[0], RssEntry)
    assert entries[0].id == 1
    assert entries[0].title == "T"
    assert entries[0].link == "http://x"
    assert entries[0].content == "<p>c</p>"
    assert entries[0].published_at == "2026-07-16"
    assert entries[0].source == "F"


def _entry_json(
    entry_id: int,
    *,
    url: str | None = None,
    title: str = "T",
    content: str = "<p>c</p>",
    published_at: str = "2026-07-16",
) -> dict:
    return {
        "id": entry_id,
        "title": title,
        "url": url if url is not None else f"http://x/{entry_id}",
        "content": content,
        "published_at": published_at,
        "feed": {"title": "F"},
    }


async def test_get_entries_paginates_across_full_pages(make_client):
    # Arrange: max_page_limit=2, requested limit=4 -> two full pages of 2
    handler, calls = _sequence_handler(
        [
            httpx.Response(200, json=[{"id": 7, "title": "news"}]),
            httpx.Response(
                200,
                json={"total": 4, "entries": [_entry_json(1), _entry_json(2)]},
            ),
            httpx.Response(
                200,
                json={"total": 4, "entries": [_entry_json(3), _entry_json(4)]},
            ),
        ]
    )
    client = make_client(handler, max_page_limit=2)

    # Act
    entries = await client.get_entries(
        "news",
        published_after=1752700000,
        published_before=1752800000,
        order="published_at",
        limit=4,
    )

    # Assert
    assert len(calls) == 3
    assert calls[1].url.params["limit"] == "2"
    assert calls[1].url.params["offset"] == "0"
    assert calls[2].url.params["limit"] == "2"
    assert calls[2].url.params["offset"] == "2"
    assert [e.id for e in entries] == [1, 2, 3, 4]


async def test_get_entries_stops_when_page_shorter_than_requested(make_client):
    # Arrange: max_page_limit=2, limit=10, server exhausted after 3 entries
    handler, calls = _sequence_handler(
        [
            httpx.Response(200, json=[{"id": 7, "title": "news"}]),
            httpx.Response(
                200,
                json={"total": 3, "entries": [_entry_json(1), _entry_json(2)]},
            ),
            httpx.Response(
                200,
                json={"total": 3, "entries": [_entry_json(3)]},
            ),
        ]
    )
    client = make_client(handler, max_page_limit=2)

    # Act
    entries = await client.get_entries(
        "news",
        published_after=1752700000,
        published_before=1752800000,
        order="published_at",
        limit=10,
    )

    # Assert: loop stops after short page, no third entries request made
    assert len(calls) == 3
    assert [e.id for e in entries] == [1, 2, 3]


def _at(entry_id: int, url: str, hour: int) -> dict:
    return _entry_json(
        entry_id, url=url, published_at=f"2026-07-16T{hour:02d}:00:00Z"
    )


def _duplicate_page() -> list[dict]:
    # same url captured three times, out of chronological order
    return [
        _entry_json(
            1,
            url="http://x/a",
            title="first",
            content="oldest",
            published_at="2026-07-16T01:00:00Z",
        ),
        _entry_json(
            2,
            url="http://x/a",
            title="third",
            content="newest",
            published_at="2026-07-16T03:00:00Z",
        ),
        _entry_json(
            3,
            url="http://x/a",
            title="second",
            content="middle",
            published_at="2026-07-16T02:00:00Z",
        ),
        _at(4, "http://x/b", 4),
    ]


async def _fetch(client: MinifluxClient, limit: int = 10) -> list[RssEntry]:
    return await client.get_entries(
        "news",
        published_after=1752700000,
        published_before=1752800000,
        order="published_at",
        limit=limit,
    )


def _entries_handler(
    pages: list[list[dict]],
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    return _sequence_handler(
        [
            httpx.Response(200, json=[{"id": 7, "title": "news"}]),
            *(
                httpx.Response(200, json={"total": len(page), "entries": page})
                for page in pages
            ),
        ]
    )


async def test_get_entries_collapses_duplicate_links(make_client):
    # Arrange
    handler, _ = _entries_handler([_duplicate_page()])
    client = make_client(handler)

    # Act
    entries = await _fetch(client)

    # Assert
    assert [entry.link for entry in entries] == ["http://x/a", "http://x/b"]


async def test_get_entries_keeps_latest_content_and_earliest_published_at(
    make_client,
):
    # Arrange
    handler, _ = _entries_handler([_duplicate_page()])
    client = make_client(handler)

    # Act
    entries = await _fetch(client)

    # Assert: newest capture carries the content, oldest carries the time
    merged = entries[0]
    assert merged.id == 2
    assert merged.title == "third"
    assert merged.content == "newest"
    assert merged.published_at == "2026-07-16T01:00:00Z"


async def test_get_entries_collapses_duplicates_across_pages(make_client):
    # Arrange: max_page_limit=2, limit=4 -> two full pages, no third request
    handler, calls = _entries_handler(
        [
            [_at(1, "http://x/a", 1), _at(2, "http://x/a", 2)],
            [_at(3, "http://x/a", 3), _at(4, "http://x/b", 4)],
        ]
    )
    client = make_client(handler, max_page_limit=2)

    # Act
    entries = await _fetch(client, limit=4)

    # Assert: dedup runs after pagination, it does not refill the limit
    assert len(calls) == 3
    assert [entry.id for entry in entries] == [3, 4]
    assert entries[0].published_at == "2026-07-16T01:00:00Z"


async def test_get_entries_preserves_first_occurrence_order(make_client):
    # Arrange
    handler, _ = _entries_handler(
        [
            [
                _at(1, "http://x/a", 1),
                _at(2, "http://x/b", 2),
                _at(3, "http://x/a", 3),
                _at(4, "http://x/c", 4),
            ]
        ]
    )
    client = make_client(handler)

    # Act
    entries = await _fetch(client)

    # Assert
    assert [entry.link for entry in entries] == [
        "http://x/a",
        "http://x/b",
        "http://x/c",
    ]


async def test_get_entries_collapses_links_differing_by_trailing_slash(
    make_client,
):
    # Arrange
    handler, _ = _entries_handler(
        [[_at(1, "http://x/a", 1), _at(2, "http://x/a/", 2)]]
    )
    client = make_client(handler)

    # Act
    entries = await _fetch(client)

    # Assert
    assert len(entries) == 1


async def test_get_entries_raises_on_invalid_entry(make_client):
    # Arrange
    handler, _ = _sequence_handler(
        [
            httpx.Response(200, json=[{"id": 7, "title": "news"}]),
            httpx.Response(
                200,
                json={
                    "total": 1,
                    "entries": [
                        {
                            "id": "not-a-number",
                            "title": "T",
                            "url": "http://x",
                            "content": "<p>c</p>",
                            "published_at": "2026-07-16",
                            "feed": {"title": "F"},
                        }
                    ],
                },
            ),
        ]
    )
    client = make_client(handler)

    # Act / Assert
    with pytest.raises(pydantic.ValidationError):
        await client.get_entries(
            "news",
            published_after=1752700000,
            published_before=1752800000,
            order="published_at",
            limit=10000,
        )


async def test_get_entries_propagates_unknown_category(make_client):
    # Arrange
    handler, _ = _constant_handler(
        httpx.Response(200, json=[{"id": 3, "title": "tech"}])
    )
    client = make_client(handler)

    # Act / Assert
    with pytest.raises(LookupError):
        await client.get_entries(
            "news",
            published_after=1752700000,
            published_before=1752800000,
            order="published_at",
            limit=10000,
        )


async def test_retries_on_transient_http_error_then_succeeds(make_client):
    # Arrange
    handler, calls = _sequence_handler(
        [
            httpx.ConnectError("boom"),
            httpx.ConnectError("boom"),
            httpx.Response(200, json=[{"id": 7, "title": "news"}]),
        ]
    )
    client = make_client(handler, attempts=3)

    # Act
    category_id = await client.get_category_id("news")

    # Assert
    assert category_id == 7
    assert len(calls) == 3


async def test_gives_up_after_retry_attempts(make_client):
    # Arrange
    handler, calls = _constant_handler(httpx.ConnectError("boom"))
    client = make_client(handler, attempts=3)

    # Act / Assert
    with pytest.raises(httpx.ConnectError):
        await client.get_category_id("news")
    assert len(calls) == 3


async def test_http_4xx_status_raises_without_retry(make_client):
    # Arrange
    handler, calls = _constant_handler(
        httpx.Response(401, json={"error": "unauthorized"})
    )
    client = make_client(handler, attempts=3)

    # Act / Assert
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_category_id("news")
    assert len(calls) == 1


async def test_http_503_status_is_retried(make_client):
    # Arrange
    handler, calls = _sequence_handler(
        [
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json=[{"id": 7, "title": "news"}]),
        ]
    )
    client = make_client(handler, attempts=3)

    # Act
    category_id = await client.get_category_id("news")

    # Assert
    assert category_id == 7
    assert len(calls) == 3


async def test_miniflux_client_factory_configures_and_closes_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
):
    # Arrange
    http_client = type("FakeHttpClient", (), {"headers": {}, "aclose": AsyncMock()})()
    monkeypatch.setattr(
        "news.digest.miniflux_client.httpx.AsyncClient", lambda **_: http_client
    )
    settings = Settings(
        miniflux_api_base="http://miniflux.test",
        miniflux_api_key="secret",
        litellm_api_key="unused",
        litellm_router="http://router.test",
        digest_output_dir=tmp_path,
    )

    # Act
    async with miniflux_client(settings) as client:
        # Assert
        assert client.base_url == "http://miniflux.test"
        assert http_client.headers["X-Auth-Token"] == "secret"

    # Assert
    http_client.aclose.assert_awaited_once()


async def test_miniflux_client_factory_closes_client_when_context_body_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
):
    # Arrange
    http_client = type("FakeHttpClient", (), {"headers": {}, "aclose": AsyncMock()})()
    monkeypatch.setattr(
        "news.digest.miniflux_client.httpx.AsyncClient", lambda **_: http_client
    )
    settings = Settings(
        miniflux_api_base="http://miniflux.test",
        miniflux_api_key="secret",
        litellm_api_key="unused",
        litellm_router="http://router.test",
        digest_output_dir=tmp_path,
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="boom"):
        async with miniflux_client(settings):
            raise RuntimeError("boom")

    http_client.aclose.assert_awaited_once()

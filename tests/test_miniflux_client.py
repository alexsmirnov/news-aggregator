from collections.abc import Callable
from unittest.mock import AsyncMock

import httpx
import pytest

from news.digest.miniflux_client import MinifluxClient, miniflux_client
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
            api_key="secret",
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


async def test_get_entries_sends_params_and_returns_entries(make_client):
    # Arrange
    handler, calls = _constant_handler(
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
        )
    )
    client = make_client(handler)

    # Act
    entries = await client.get_entries(
        7, published_after=1752700000, order="published_at", limit=10000
    )

    # Assert
    assert calls[0].url.path == "/v1/categories/7/entries"
    assert calls[0].url.params["published_after"] == "1752700000"
    assert calls[0].url.params["order"] == "published_at"
    assert calls[0].url.params["limit"] == "10000"
    assert len(entries) == 1
    assert entries[0]["feed"]["title"] == "F"


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

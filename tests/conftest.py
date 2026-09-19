import re
import types
from collections.abc import Callable, Generator
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from news.digest.archive import digest_path
from news.digest.schemas import Digest, RssEntry
from news.pages import get_settings
from news.server import app
from news.settings import Settings


def body_text(html: str) -> str:
    body = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL)
    assert body is not None
    text = re.sub(r"<[^>]+>", "", body.group(1))
    return " ".join(text.split())


def title_text(html: str) -> str:
    title = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
    assert title is not None
    return title.group(1).strip()


@pytest.fixture
def client(tmp_path: Path) -> Generator[TestClient]:
    # ponytail: bypasses scheduler_lifespan (which needs real Miniflux/
    # LLM credentials) by overriding only the settings dependency the
    # page routes actually use.
    settings = Settings(
        miniflux_api_base="http://m.test",  # type: ignore[arg-type]
        miniflux_api_key="k",  # type: ignore[arg-type]
        litellm_api_key="l",  # type: ignore[arg-type]
        litellm_router="http://r.test",  # type: ignore[arg-type]
        digest_output_dir=tmp_path,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


SeedDigest = Callable[[str, date, Digest], Path]


@pytest.fixture
def seed_digest(tmp_path: Path) -> SeedDigest:
    def _seed(name: str, day: date, digest: Digest) -> Path:
        path = digest_path(tmp_path, name, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(digest.model_dump_json())
        return path

    return _seed


class FakeMiniflux:
    def __init__(
        self,
        entries: list[RssEntry] | None = None,
        category_error: Exception | None = None,
        entries_error: Exception | None = None,
    ) -> None:
        self.entries = entries if entries is not None else []
        self.category_error = category_error
        self.entries_error = entries_error
        self.calls: list[dict[str, Any]] = []

    async def get_entries(
        self, category_name: str, **kwargs: Any
    ) -> list[RssEntry]:
        self.calls.append({"category_name": category_name, **kwargs})
        if self.category_error is not None:
            raise self.category_error
        if self.entries_error is not None:
            raise self.entries_error
        return self.entries


class FakeLlm:
    def __init__(
        self,
        chat_results: list[Any] | None = None,
        chat_parsed_results: list[Any] | None = None,
        embeddings_results: list[Any] | None = None,
    ) -> None:
        self.chat_results = list(chat_results or [])
        self.chat_parsed_results = list(chat_parsed_results or [])
        self.embeddings_results = list(embeddings_results or [])
        self.chat_calls: list[tuple[Any, ...]] = []
        self.chat_parsed_calls: list[tuple[Any, ...]] = []
        self.embeddings_calls: list[tuple[Any, ...]] = []

    async def chat(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        self.chat_calls.append((model, messages, kwargs))
        result = self.chat_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def chat_parsed(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_format: Any,
        **kwargs: Any,
    ) -> Any:
        self.chat_parsed_calls.append(
            (model, messages, response_format, kwargs)
        )
        result = self.chat_parsed_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def embeddings(
        self, model: str, inputs: list[str], **kwargs: Any
    ) -> Any:
        self.embeddings_calls.append((model, inputs, kwargs))
        result = self.embeddings_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def entry() -> RssEntry:
    return RssEntry(
        id=42,
        title="T",
        content="C",
        link="http://a",
        published_at="2026-07-16T10:00:00",
        source="F",
    )


@pytest.fixture
def fake_miniflux() -> type[FakeMiniflux]:
    return FakeMiniflux


@pytest.fixture
def fake_llm() -> type[FakeLlm]:
    return FakeLlm


@pytest.fixture
def settings_stub(tmp_path: Path) -> Settings:
    return cast(
        Settings,
        types.SimpleNamespace(
            miniflux_api_base="http://m.test",
            miniflux_api_key="k",
            litellm_api_key="l",
            litellm_router="http://r.test",
            digest_output_dir=tmp_path,
            fetch_lookback_hours=24,
            fetch_limit=10000,
            entry_content_max_chars=1000,
            grouping_content_max_chars=300,
            grouping_k_sigma=3.0,
            grouping_map_clusters=40,
            grouping_max_records=20,
            refine_max_links=10,
            model_trending="sonar-reasoning-pro",
            model_grouping="gemini-flash",
            model_refinement="gemini-flash",
            model_embedding="bge-embed",
            embedding_dimensions=1024,
            schedule_timezone="America/Los_Angeles",
        ),
    )

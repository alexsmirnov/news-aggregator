import re
from collections.abc import Callable, Generator
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from news.digest.archive import digest_path
from news.digest.schemas import Digest
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

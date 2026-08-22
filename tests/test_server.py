from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import body_text, title_text

import news.server

_ = (body_text, title_text)  # re-exported for tests in this module


class FakeJob:
    def __init__(self, next_run_time: datetime) -> None:
        self.id = "news_digest"
        self.pending = False
        self.next_run_time = next_run_time

    def modify(self, **changes: datetime) -> None:
        self.next_run_time = changes["next_run_time"]


class FakeScheduler:
    def __init__(self, job: FakeJob | None, *, running: bool = True) -> None:
        self.job = job
        self.running = running

    def get_job(self, job_id: str) -> FakeJob | None:
        return self.job if job_id == "news_digest" else None


class FixedDatetime(datetime):
    current: datetime

    @classmethod
    def now(cls, tz: object = None) -> datetime:
        return cls.current


@pytest.fixture
def scheduler_job() -> Generator[FakeJob]:
    job = FakeJob(datetime(2026, 8, 23, 12, tzinfo=UTC))
    news.server.app.state.scheduler = FakeScheduler(job)
    yield job
    del news.server.app.state.scheduler


@pytest.fixture
def set_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIFLUX_API_BASE", "http://m.test")
    monkeypatch.setenv("MINIFLUX_API_KEY", "k")
    monkeypatch.setenv("LITELLM_API_KEY", "l")
    monkeypatch.setenv("LITELLM_ROUTER", "http://r.test")
    monkeypatch.setenv("DIGEST_OUTPUT_DIR", str(tmp_path))


async def test_run_aggregate_builds_service_from_settings_and_calls_it(
    set_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    constructed: dict[str, object] = {}

    class FakeDigestService:
        def __init__(
            self, settings: object, miniflux: object, llm: object
        ) -> None:
            constructed["settings"] = settings
            constructed["miniflux"] = miniflux
            constructed["llm"] = llm
            constructed["called"] = False

        async def __call__(self) -> list[object]:
            constructed["called"] = True
            return []

    monkeypatch.setattr(news.server, "DigestService", FakeDigestService)

    # Act
    await news.server.run_aggregate()

    # Assert
    assert constructed["settings"] is not None
    assert constructed["miniflux"] is not None
    assert constructed["llm"] is not None
    assert constructed["called"] is True


def test_get_aggregate_returns_scheduler_job_status(
    client: TestClient, scheduler_job: FakeJob
) -> None:
    # Act
    response = client.get("/aggregate")

    # Assert
    assert response.status_code == 200
    assert response.json() == {
        "job_id": "news_digest",
        "scheduler_running": True,
        "job_pending": False,
        "next_run_time": "2026-08-23T12:00:00Z",
    }


def test_post_aggregate_schedules_job_immediately(
    client: TestClient,
    scheduler_job: FakeJob,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    now = datetime(2026, 8, 23, 13, tzinfo=UTC)
    FixedDatetime.current = now
    monkeypatch.setattr("news.aggregate.datetime", FixedDatetime)

    # Act
    response = client.post("/aggregate")

    # Assert
    assert response.status_code == 202
    assert scheduler_job.next_run_time == now
    assert response.json()["next_run_time"] == "2026-08-23T13:00:00Z"


@pytest.mark.parametrize("method", ["get", "post"])
def test_aggregate_returns_unavailable_when_scheduler_job_is_missing(
    client: TestClient, scheduler_job: FakeJob, method: str
) -> None:
    # Arrange
    news.server.app.state.scheduler = FakeScheduler(None)

    # Act
    response = getattr(client, method)("/aggregate")

    # Assert
    assert response.status_code == 503


def test_post_aggregate_returns_unavailable_when_scheduler_is_stopped(
    client: TestClient, scheduler_job: FakeJob
) -> None:
    # Arrange
    news.server.app.state.scheduler = FakeScheduler(
        scheduler_job, running=False
    )

    # Act
    response = client.post("/aggregate")

    # Assert
    assert response.status_code == 503


def test_home_page_returns_ok_html_without_redirect(
    client: TestClient,
) -> None:
    # Arrange
    path = "/"

    # Act
    response = client.get(path, follow_redirects=False)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_home_page_title_and_body_text(client: TestClient) -> None:
    # Arrange
    path = "/"

    # Act
    response = client.get(path)

    # Assert
    assert title_text(response.text) == "News Aggregator"
    assert (
        body_text(response.text)
        == "news economy technology No digests available yet"
    )


def test_home_page_references_shared_stylesheet_and_favicon(
    client: TestClient,
) -> None:
    # Arrange
    path = "/"

    # Act
    response = client.get(path)

    # Assert
    assert 'href="/style.css"' in response.text
    assert 'href="/favicon.svg"' in response.text


def test_home_page_loads_htmx_from_base_layout(client: TestClient) -> None:
    # Arrange
    path = "/"

    # Act
    response = client.get(path)

    # Assert
    assert (
        "https://cdn.jsdelivr.net/npm/htmx.org@2.0.10/dist/htmx.min.js"
        in response.text
    )
    assert (
        'integrity="sha384-H5SrcfygHmAuTDZphMHqBJLc3FhssKjG7w/'
        'CeCpFReSfwBWDTKpkzPP8c+cLsK+V"' in response.text
    )
    assert 'crossorigin="anonymous"' in response.text
    assert (
        body_text(response.text)
        == "news economy technology No digests available yet"
    )


def test_home_page_responds_to_head_request(client: TestClient) -> None:
    # Arrange
    path = "/"

    # Act
    response = client.head(path, follow_redirects=False)

    # Assert
    assert response.status_code == 200


def test_home_page_declares_doctype_and_language(
    client: TestClient,
) -> None:
    # Arrange
    path = "/"

    # Act
    response = client.get(path)

    # Assert
    assert response.text.lstrip().lower().startswith("<!doctype html>")
    assert 'lang="en"' in response.text


def test_home_page_is_not_in_openapi_schema(client: TestClient) -> None:
    # Arrange
    path = "/openapi.json"

    # Act
    response = client.get(path)

    # Assert
    assert "/" not in response.json()["paths"]


def test_stylesheet_loads(client: TestClient) -> None:
    # Arrange
    path = "/style.css"

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_favicon_loads(client: TestClient) -> None:
    # Arrange
    path = "/favicon.svg"

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"


def test_undefined_path_returns_404_html_without_redirect(
    client: TestClient,
) -> None:
    # Arrange
    path = "/does-not-exist"

    # Act
    response = client.get(path, follow_redirects=False)

    # Assert
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")


def test_undefined_path_title_and_body_text(client: TestClient) -> None:
    # Arrange
    path = "/does-not-exist"

    # Act
    response = client.get(path)

    # Assert
    assert title_text(response.text) == "404 - News Aggregator"
    assert body_text(response.text) == "Page Not Found"


def test_undefined_path_references_shared_stylesheet_and_favicon(
    client: TestClient,
) -> None:
    # Arrange
    path = "/does-not-exist"

    # Act
    response = client.get(path)

    # Assert
    assert 'href="/style.css"' in response.text
    assert 'href="/favicon.svg"' in response.text


def test_missing_asset_path_returns_404_html(client: TestClient) -> None:
    # Arrange
    path = "/missing.css"

    # Act
    response = client.get(path, follow_redirects=False)

    # Assert
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert title_text(response.text) == "404 - News Aggregator"


def test_docs_endpoint_still_available(client: TestClient) -> None:
    # Arrange
    path = "/docs"

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200


def test_redoc_endpoint_still_available(client: TestClient) -> None:
    # Arrange
    path = "/redoc"

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200


def test_openapi_schema_still_available(client: TestClient) -> None:
    # Arrange
    path = "/openapi.json"

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200
    assert response.json() is not None

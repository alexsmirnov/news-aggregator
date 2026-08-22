import asyncio
import inspect
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from news.digest.service import DigestService
from news.scheduler import scheduler_lifespan
from news.server import create_app

ENV_VARS = (
    "MINIFLUX_API_BASE",
    "MINIFLUX_API_KEY",
    "LITELLM_API_KEY",
    "LITELLM_ROUTER",
    "DIGEST_OUTPUT_DIR",
)


@pytest.fixture
def set_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIFLUX_API_BASE", "http://m.test")
    monkeypatch.setenv("MINIFLUX_API_KEY", "k")
    monkeypatch.setenv("LITELLM_API_KEY", "l")
    monkeypatch.setenv("LITELLM_ROUTER", "http://r.test")
    monkeypatch.setenv("DIGEST_OUTPUT_DIR", str(tmp_path))


@pytest.fixture
def clear_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


def test_create_app_does_not_require_environment(clear_env: None) -> None:
    # Act
    app = create_app()

    # Assert
    assert isinstance(app, FastAPI)


async def test_lifespan_missing_env_raises_validation_error(
    clear_env: None,
) -> None:
    # Arrange
    app = create_app()

    # Act / Assert
    with pytest.raises(ValidationError):
        async with scheduler_lifespan(app):
            pass


async def test_lifespan_starts_scheduler_with_interval_job(
    set_env: None,
) -> None:
    # Arrange
    app = create_app()

    # Act
    async with scheduler_lifespan(app):
        scheduler = app.state.scheduler
        job = scheduler.get_job("news_digest")

        # Assert
        assert scheduler.running is True
        assert job is not None
        assert job.trigger.interval == timedelta(hours=12)
        assert job.max_instances == 1
        assert job.coalesce is True
        assert inspect.iscoroutinefunction(job.func)
        assert isinstance(job.func.__self__, DigestService)
        assert job.args == ()


async def test_lifespan_shutdown_stops_scheduler(set_env: None) -> None:
    # Arrange
    app = create_app()

    # Act
    async with scheduler_lifespan(app):
        scheduler = app.state.scheduler
    await asyncio.sleep(0)

    # Assert
    assert scheduler.running is False


def test_app_boot_runs_lifespan_via_testclient(set_env: None) -> None:
    # Arrange
    app = create_app()

    # Act
    with TestClient(app) as client:
        response = client.get("/")

        # Assert
        assert response.status_code == 200
        assert app.state.scheduler.running is True

    assert app.state.scheduler.running is False

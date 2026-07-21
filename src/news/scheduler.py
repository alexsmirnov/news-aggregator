from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI

from news.digest.llm_client import LlmClient, llm_client
from news.digest.miniflux_client import MinifluxClient, miniflux_client
from news.digest.service import DigestService
from news.settings import Settings


def build_scheduler(
    settings: Settings, miniflux: MinifluxClient, llm: LlmClient
) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()
    service = DigestService(settings, miniflux, llm)
    scheduler.add_job(
        service,
        IntervalTrigger(hours=settings.schedule_interval_hours),
        id="news_digest",
        max_instances=1,
        coalesce=True,
    )
    return scheduler


@asynccontextmanager
async def scheduler_lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = Settings() # type: ignore BaseSettings fill from environment
    async with (
        miniflux_client(settings) as miniflux,
        llm_client(settings) as llm,
    ):
        scheduler = build_scheduler(settings, miniflux, llm)
        scheduler.start()
        app.state.scheduler = scheduler
        try:
            yield
        finally:
            scheduler.shutdown(wait=True)

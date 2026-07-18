import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI

from news.digest.llm_client import LlmClient
from news.digest.miniflux_client import MinifluxClient
from news.digest.service import run_all_aggregations
from news.settings import Settings

logger = logging.getLogger(__name__)


async def _run_pipeline_job(
    settings: Settings, miniflux: MinifluxClient, llm: LlmClient
) -> None:
    try:
        paths = await run_all_aggregations(settings, miniflux, llm)
        logger.info("news digests written: %s", paths)
    except Exception:
        logger.exception("news digest pipeline failed")


def build_scheduler(
    settings: Settings, miniflux: MinifluxClient, llm: LlmClient
) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _run_pipeline_job,
        IntervalTrigger(hours=settings.schedule_interval_hours),
        args=[settings, miniflux, llm],
        id="news_digest",
        max_instances=1,
        coalesce=True,
    )
    return scheduler


@asynccontextmanager
async def scheduler_lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = Settings()
    miniflux: MinifluxClient | None = None
    llm: LlmClient | None = None
    scheduler: AsyncIOScheduler | None = None
    try:
        miniflux = MinifluxClient(
            str(settings.miniflux_api_base), settings.miniflux_api_key
        )
        llm = LlmClient(settings.litellm_api_key, str(settings.litellm_router))
        scheduler = build_scheduler(settings, miniflux, llm)
        scheduler.start()
        app.state.scheduler = scheduler
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)
            # AsyncIOScheduler defers the actual state change to the next
            # event-loop iteration via call_soon_threadsafe; without this
            # yield, scheduler.running can still read True right after
            # shutdown() returns.
            await asyncio.sleep(0)
        if miniflux is not None:
            await miniflux.aclose()
        if llm is not None:
            await llm.aclose()

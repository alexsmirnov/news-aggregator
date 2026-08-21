from pathlib import Path

from fastapi import FastAPI

from news.digest.llm_client import llm_client
from news.digest.miniflux_client import miniflux_client
from news.digest.service import DigestService
from news.pages import router as pages_router
from news.scheduler import scheduler_lifespan
from news.settings import Settings


def create_app() -> FastAPI:
    """Create the News Aggregator web application."""
    app = FastAPI(lifespan=scheduler_lifespan)
    app.include_router(pages_router)
    frontend_dir = Path(__file__).resolve().parent / "frontend"
    app.frontend("/", directory=frontend_dir, fallback="404.html")
    return app


async def run_aggregate() -> None:
    settings = Settings()  # type: ignore BaseSettings fill from environment
    async with (
        miniflux_client(settings) as miniflux,
        llm_client(settings) as llm,
    ):
        await DigestService(settings, miniflux, llm)()


app = create_app()

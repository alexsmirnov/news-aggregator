from pathlib import Path

from fastapi import FastAPI

from news.scheduler import scheduler_lifespan


def create_app() -> FastAPI:
    """Create the News Aggregator web application."""
    app = FastAPI(lifespan=scheduler_lifespan)
    frontend_dir = Path(__file__).resolve().parent / "frontend"
    app.frontend("/", directory=frontend_dir, fallback="404.html")
    return app


app = create_app()

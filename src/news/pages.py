from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, Response
from fastapi.templating import Jinja2Templates

from news.digest.archive import build_page
from news.settings import Settings

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore BaseSettings fill from environment


def _not_found() -> FileResponse:
    return FileResponse(
        FRONTEND_DIR / "404.html",
        status_code=404,
        media_type="text/html",
    )


def _render_digest(
    request: Request,
    settings: Settings,
    name: str | None,
    day: date | None,
) -> Response:
    page = build_page(
        settings.digest_output_dir, settings.aggregations, name, day
    )
    if page is None:
        return _not_found()
    return templates.TemplateResponse(request, "digest.html", {"page": page})


@router.api_route(
    "/",
    methods=["GET", "HEAD"],
    name="home",
    include_in_schema=False,
)
def home(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Render the most recent digest for the default aggregation."""
    return _render_digest(request, settings, None, None)


@router.api_route(
    "/digest/{name}",
    methods=["GET", "HEAD"],
    include_in_schema=False,
)
def digest_by_name(
    request: Request,
    name: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Render the most recent digest for the given aggregation."""
    return _render_digest(request, settings, name, None)


@router.api_route(
    "/digest/{name}/{day}",
    methods=["GET", "HEAD"],
    include_in_schema=False,
)
def digest_by_date(
    request: Request,
    name: str,
    day: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Render the digest for the given aggregation and ISO date."""
    try:
        parsed_day = date.fromisoformat(day)
    except ValueError:
        return _not_found()
    return _render_digest(request, settings, name, parsed_day)

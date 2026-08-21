from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter()


@router.api_route(
    "/",
    methods=["GET", "HEAD"],
    response_class=HTMLResponse,
    name="home",
    include_in_schema=False,
)
async def home(request: Request) -> HTMLResponse:
    """Serve the home page rendered from the Jinja2 template."""
    return templates.TemplateResponse(request, "index.html")

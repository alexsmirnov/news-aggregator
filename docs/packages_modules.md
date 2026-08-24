# Packages & Modules

The project is a single Python package `news` (installed from [pyproject.toml](../pyproject.toml), CLI entry point `news = "news:main"`) with one sub-package `news.digest`. Source lives in `src/news/`, tests in `tests/`. #packages #modules

## news #module

Top-level application package: CLI entry point, FastAPI app assembly, HTTP routes, scheduling, and settings. Source: [src/news/](../src/news/)

**Responsibility:** parse CLI commands and run Uvicorn or a one-shot aggregation ([src/news/__init__.py:66-97](../src/news/__init__.py#L66-L97)); build the FastAPI app with lifespan, routers, and static frontend ([src/news/server.py:13-19](../src/news/server.py#L13-L19)); expose aggregate control endpoints ([src/news/aggregate.py](../src/news/aggregate.py)) and digest page routes ([src/news/pages.py](../src/news/pages.py)); manage the APScheduler job lifecycle ([src/news/scheduler.py](../src/news/scheduler.py)); define settings, default aggregations, and topic focus texts ([src/news/settings.py](../src/news/settings.py)).

Modules:

| Module | Responsibility |
|---|---|
| `__init__.py` | CLI: `server` and `aggregate` subcommands, logging configuration, Uvicorn launch ([src/news/__init__.py:10-97](../src/news/__init__.py#L10-L97)) |
| `server.py` | `create_app()` app factory and `run_aggregate()` one-shot pipeline runner ([src/news/server.py:13-28](../src/news/server.py#L13-L28)) |
| `scheduler.py` | `build_scheduler()` interval job and `scheduler_lifespan` FastAPI lifespan ([src/news/scheduler.py:16-44](../src/news/scheduler.py#L16-L44)) |
| `aggregate.py` | `/aggregate` GET/POST router and `AggregateStatus` model ([src/news/aggregate.py:11-63](../src/news/aggregate.py#L11-L63)) |
| `pages.py` | `/` and `/digest/...` page routes, Jinja2 templates setup, settings dependency ([src/news/pages.py:17-91](../src/news/pages.py#L17-L91)) |
| `settings.py` | `Settings` pydantic-settings model, `Aggregation` model, focus texts, default aggregations ([src/news/settings.py:8-69](../src/news/settings.py#L8-L69)) |

Presentation assets (not Python modules): Jinja2 page templates in [src/news/templates/](../src/news/templates/) (`base.html` layout, `digest.html` digest page) and static frontend files in [src/news/frontend/](../src/news/frontend/) (`style.css`, `favicon.svg`, `404.html`).

- **Uses:** `news.digest`
- **Used by:** — (application root)

## news.digest #module

Digest pipeline sub-package: fetches RSS entries from Miniflux, groups and refines news with LLM calls, persists digest JSON, and reads the archive back into page view models. Source: [src/news/digest/](../src/news/digest/)

| Module | Responsibility |
|---|---|
| `service.py` | `DigestService` pipeline: fetch, HTML strip, entry formatting, LLM grouping, concurrent refinement, digest writing, per-aggregation orchestration; `PipelineError` ([src/news/digest/service.py:29-390](../src/news/digest/service.py#L29-L390)) |
| `miniflux_client.py` | `MinifluxClient` HTTP client: category lookup, paginated entry fetching, auth header, transient-error retries; `miniflux_client` factory context manager ([src/news/digest/miniflux_client.py:21-224](../src/news/digest/miniflux_client.py#L21-L224)) |
| `llm_client.py` | `LlmClient` OpenAI-compatible wrapper: `chat` (raw text) and `chat_parsed` (structured output) with retries; `llm_client` factory context manager ([src/news/digest/llm_client.py:24-138](../src/news/digest/llm_client.py#L24-L138)) |
| `archive.py` | Read-only archive reader: digest path layout, available dates, digest loading, link sanitization, `build_page` view-model builder ([src/news/digest/archive.py:27-152](../src/news/digest/archive.py#L27-L152)) |
| `schemas.py` | Pydantic models: `RssEntry`, `NewsRecord`, `NewsResponse`, `DigestRecord`, `Digest`, `NavLink`, `RecordView`, `DigestPage` ([src/news/digest/schemas.py:6-54](../src/news/digest/schemas.py#L6-L54)) |
| `prompts.py` | LLM prompt builders: trending query, grouping system/user prompts, refinement system/user prompts ([src/news/digest/prompts.py](../src/news/digest/prompts.py)) |

- **Uses:** `news` (settings)
- **Used by:** `news` (server, scheduler, pages)

See [Architecture Overview](architecture_overview.md) for the end-to-end data flow and [Tests](tests_coverage.md) for the corresponding test suites.

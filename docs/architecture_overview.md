# Architecture Overview

The Breaking News Aggregator is a FastAPI application that periodically fetches RSS entries from Miniflux, uses LLM calls to group trending news and write per-topic summaries, persists digests as JSON files, and serves them as HTML pages. #architecture #design

## Technology Stack #architecture

- **Language/runtime:** Python 3.14 ([pyproject.toml:9](../pyproject.toml#L9)), dependencies managed with uv
- **Web framework:** FastAPI ([src/news/server.py:13-19](../src/news/server.py#L13-L19)), served by Uvicorn ([src/news/__init__.py:91-97](../src/news/__init__.py#L91-L97))
- **Scheduling:** APScheduler `AsyncIOScheduler` running inside the server process ([src/news/scheduler.py:16-28](../src/news/scheduler.py#L16-L28))
- **AI calls:** OpenAI-compatible router (LiteLLM) via the official `openai` SDK ([src/news/digest/llm_client.py:40-42](../src/news/digest/llm_client.py#L40-L42))
- **News source:** Miniflux feed aggregator HTTP API ([src/news/digest/miniflux_client.py:28-52](../src/news/digest/miniflux_client.py#L28-L52))
- **Frontend:** plain HTML from Jinja2 templates ([src/news/templates/](../src/news/templates/)), plain CSS ([src/news/frontend/style.css](../src/news/frontend/style.css)), htmx loaded from CDN in the base layout ([src/news/templates/base.html:9](../src/news/templates/base.html#L9))
- **Storage:** filesystem JSON files, no application database ([src/news/digest/archive.py:27-33](../src/news/digest/archive.py#L27-L33))
- **Deployment:** Docker container via Docker Compose with Miniflux and PostgreSQL ([docker-compose.yml](../docker-compose.yml))

## System Data Flow #architecture

```
Miniflux (RSS)  ──fetch──▶  DigestService  ──group (LLM)──▶  refine (LLM + url_context)
                                     │
                                     ▼
                    digest_output_dir/YYYY/MM/<name>-DD.json
                                     │
                       archive.build_page (read + view model)
                                     │
                                     ▼
                    Jinja2 digest.html → browser (htmx-ready)
```

1. **Fetch:** `DigestService.fetch_entries` resolves the Miniflux category and paginates entries published within the lookback window, strips HTML with BeautifulSoup, and truncates content ([src/news/digest/service.py:64-102](../src/news/digest/service.py#L64-L102)). Miniflux re-captures an article whenever its page is updated, emitting a new entry id and `published_at` for the same url, so `MinifluxClient.get_entries` collapses entries sharing a link (trailing slash ignored) into one: the newest capture's title and content with the earliest capture's `published_at`, keeping breaking-news timing intact ([src/news/digest/miniflux_client.py:28-53](../src/news/digest/miniflux_client.py#L28-L53)). Deduplication runs after pagination stops, so `limit` is an upper bound on the returned count.
2. **Group:** `DigestService` delegates to an injected `Grouping` collaborator ([src/news/digest/grouping.py](../src/news/digest/grouping.py)); the default `LlmGrouping` sends formatted entries to the grouping model as structured output (`response_format=NewsResponse`, `reasoning_effort="high"`), producing `NewsRecord` items of a combined headline plus source links (prompts in [src/news/digest/prompts.py:8-33](../src/news/digest/prompts.py#L8-L33)). The `Grouping` protocol makes alternative implementations swappable without changing `DigestService`.
3. **Refine:** for each group, matched entry content plus up to `refine_max_links` links are summarized by the refinement model with the `url_context` tool so the model can read linked pages directly; refinements run concurrently capped at 8 ([src/news/digest/service.py:104-233](../src/news/digest/service.py#L104-L233)).
4. **Persist:** the `Digest` JSON is written to `<digest_output_dir>/YYYY/MM/<name>-DD.json` ([src/news/digest/service.py:234-256](../src/news/digest/service.py#L234-L256), path layout in [src/news/digest/archive.py:27-33](../src/news/digest/archive.py#L27-L33)).
5. **Render:** page routes load digests from disk via `archive.build_page`, which builds a `DigestPage` view model with aggregation navigation, older/newer date links, and sanitized records (only http/https links) ([src/news/digest/archive.py:93-152](../src/news/digest/archive.py#L93-L152), rendered by [src/news/templates/digest.html](../src/news/templates/digest.html)).

Aggregations run sequentially per pipeline run; a failing aggregation is logged and skipped without aborting the others ([src/news/digest/service.py:299-318](../src/news/digest/service.py#L299-L318)). Default aggregations map names to Miniflux categories with topic focus texts: `news`, `economy`, `technology` ([src/news/settings.py:39-43](../src/news/settings.py#L39-L43)).

## Scheduling #architecture

A single interval job (`news_digest`) runs `DigestService.__call__` every `schedule_interval_hours` (default 12) with `max_instances=1` and `coalesce=True` ([src/news/scheduler.py:13-28](../src/news/scheduler.py#L13-L28)). The FastAPI lifespan constructs settings, opens shared Miniflux and LLM clients, starts the scheduler, and stores it on `app.state.scheduler` ([src/news/scheduler.py:31-44](../src/news/scheduler.py#L31-L44)). `POST /aggregate` reschedules the job to run immediately ([src/news/aggregate.py:46-63](../src/news/aggregate.py#L46-L63)); a one-shot run without the server is available via the `news aggregate` CLI command ([src/news/server.py:22-28](../src/news/server.py#L22-L28)).

## AI Integration #architecture #llm

`LlmClient` wraps `openai.AsyncOpenAI` pointed at the LiteLLM router base URL with 60s timeout and tenacity retries on transient errors (`APIConnectionError`, `RateLimitError`, `InternalServerError`) ([src/news/digest/llm_client.py:24-57](../src/news/digest/llm_client.py#L24-L57)). Two call modes exist: `chat` for raw text used by refinement ([src/news/digest/llm_client.py:62-89](../src/news/digest/llm_client.py#L62-L89)) and `chat_parsed` for Pydantic structured output used by grouping ([src/news/digest/llm_client.py:91-129](../src/news/digest/llm_client.py#L91-L129)). The Miniflux client applies the same retry pattern to transient HTTP errors (429, 5xx, request errors) ([src/news/digest/miniflux_client.py:21-52](../src/news/digest/miniflux_client.py#L21-L52)). Model names are configurable per role: trending, grouping, refinement, and an evaluation judge model ([src/news/settings.py:62-65](../src/news/settings.py#L62-L65)). `LlmClient.embeddings` calls the OpenAI-compatible embeddings endpoint for a not-yet-wired-in map-reduce grouping design ([src/news/digest/llm_client.py](../src/news/digest/llm_client.py)); `MapReduceGrouping.embed_entries` computes separate title and content vectors per entry, `calibrate_threshold` and `cluster` implement corpus-calibrated average-linkage clustering (scikit-learn `AgglomerativeClustering`, no noise class — singletons survive as one-element groups), but grouping itself still raises `NotImplementedError` and `LlmGrouping` remains the only implementation used by `DigestService` ([src/news/digest/map_reduce.py](../src/news/digest/map_reduce.py)).

## Layering #architecture

HTTP layer (`server.py`, `aggregate.py`, `pages.py`) → archive reader (`archive.py`) → pipeline service (`service.py`, delegating grouping to `grouping.py`) → external clients (`miniflux_client.py`, `llm_client.py`), with settings centralized in `settings.py`. The application was ported from the prototype notebook [docs/ai-news-digest.ipynb](ai-news-digest.ipynb). See [Packages & Modules](packages_modules.md) for package-level details and [API Endpoints](api_endpoints.md) for exposed routes.

# Dependencies & Libraries

External dependencies declared in [pyproject.toml](../pyproject.toml): runtime dependencies at [pyproject.toml:10-21](../pyproject.toml#L10-L21), development group at [pyproject.toml:23-33](../pyproject.toml#L23-L33). Managed with uv (`uv sync`, locked in `uv.lock`). #dependencies #libraries

## Framework #dependencies

### FastAPI `>=0.139.0`
Web framework: app factory, routers, lifespan, static frontend mount.
Docs: https://fastapi.tiangolo.com/ | Context7: `/websites/fastapi_tiangolo`
Used in: [src/news/server.py:1](../src/news/server.py#L1), [src/news/pages.py:6-8](../src/news/pages.py#L6-L8), [src/news/aggregate.py:3](../src/news/aggregate.py#L3)

### Uvicorn `>=0.51.0`
ASGI server launched by the `news server` CLI command with custom log config.
Docs: https://www.uvicorn.org/
Used in: [src/news/__init__.py:6](../src/news/__init__.py#L6), [src/news/__init__.py:91-97](../src/news/__init__.py#L91-L97)

### APScheduler `>=3.11,<4`
`AsyncIOScheduler` with interval trigger for the periodical digest job inside the server.
Docs: https://apscheduler.readthedocs.io/ | Context7: `/agronholm/apscheduler`
Used in: [src/news/scheduler.py:4-5](../src/news/scheduler.py#L4-L5), [src/news/scheduler.py:19-27](../src/news/scheduler.py#L19-L27)

### Jinja2 `>=3.1`
HTML template engine for digest pages (`base.html`, `digest.html`).
Docs: https://jinja.palletsprojects.com/
Used in: [src/news/pages.py:8](../src/news/pages.py#L8), [src/news/pages.py:15](../src/news/pages.py#L15), [src/news/templates/](../src/news/templates/)

### OpenAI Python SDK `>=2`
OpenAI-compatible async client for the LiteLLM router: chat completions and structured outputs with retries.
Docs: https://openai.github.io/openai-python/ | Context7: `/openai/openai-python`
Used in: [src/news/digest/llm_client.py:6-7](../src/news/digest/llm_client.py#L6-L7), [src/news/digest/llm_client.py:40-42](../src/news/digest/llm_client.py#L40-L42)

## Runtime #dependencies

### httpx `>=0.28`
Async HTTP client for the Miniflux API; `MockTransport` also powers its unit tests.
Docs: https://www.python-httpx.org/
Used in: [src/news/digest/miniflux_client.py:5](../src/news/digest/miniflux_client.py#L5), [tests/test_miniflux_client.py](../tests/test_miniflux_client.py)

### BeautifulSoup4 `>=4.12`
HTML stripping from RSS entry content before LLM processing.
Docs: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
Used in: [src/news/digest/service.py:7](../src/news/digest/service.py#L7), [src/news/digest/service.py:44-48](../src/news/digest/service.py#L44-L48)

### aiofiles `>=24`
Async file writing of digest JSON files.
Docs: https://github.com/Tinche/aiofiles
Used in: [src/news/digest/service.py:6](../src/news/digest/service.py#L6), [src/news/digest/service.py:313-314](../src/news/digest/service.py#L313-L314)

### pydantic-settings `>=2`
`BaseSettings` with `.env` support for all application configuration.
Docs: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
Used in: [src/news/settings.py:5](../src/news/settings.py#L5), [src/news/settings.py:46-47](../src/news/settings.py#L46-L47)

### tenacity `>=9`
Exponential-backoff retries for LLM calls and transient Miniflux HTTP errors.
Docs: https://tenacity.readthedocs.io/
Used in: [src/news/digest/llm_client.py:9-15](../src/news/digest/llm_client.py#L9-L15), [src/news/digest/miniflux_client.py:7-13](../src/news/digest/miniflux_client.py#L7-L13)

## Development #dependencies

### pytest `>=9.0.1`
Test runner; test paths, markers, and logging configured in [pyproject.toml:49-67](../pyproject.toml#L49-L67).
Docs: https://docs.pytest.org/
Used in: all files under [tests/](../tests/)

### pytest-asyncio `>=1.3.0`
Async test support with `asyncio_mode = "auto"` and module-scoped loop.
Docs: https://pytest-asyncio.readthedocs.io/
Used in: [pyproject.toml:55](../pyproject.toml#L55), [tests/evaluation/conftest.py:10](../tests/evaluation/conftest.py#L10)

### ruff `>=0.14.8`
Linter and formatter (E, W, F, I, N, UP rules; line length 79; `E501` ignored for verbatim prompt text).
Docs: https://docs.astral.sh/ruff/
Configured in: [pyproject.toml:69-92](../pyproject.toml#L69-L92)

### pyright `>=1.1.407`
Static type checker over `src` and `tests`.
Docs: https://microsoft.github.io/pyright/
Configured in: [pyproject.toml:94-99](../pyproject.toml#L94-L99)

### deepeval `>=4.1`
LLM evaluation framework: GEval metrics and GPT judge model for the evaluation suite.
Docs: https://deepeval.docs.confident-ai.com/
Used in: [tests/evaluation/test_digest_eval.py:5-8](../tests/evaluation/test_digest_eval.py#L5-L8), [tests/evaluation/conftest.py:11](../tests/evaluation/conftest.py#L11)

### rouge-score `>=0.1.2`
ROUGE-L scoring for summary quality evaluation.
Docs: https://github.com/google-research/google-research/tree/master/rouge
Used in: [tests/evaluation/metrics.py:3](../tests/evaluation/metrics.py#L3), [tests/evaluation/metrics.py:24-26](../tests/evaluation/metrics.py#L24-L26)

### datasets `>=4.5.0`
Declared in the dev dependency group; no imports in `src/` or `tests/` (evaluation fixtures use plain JSON files).
Docs: https://huggingface.co/docs/datasets

### types-aiofiles `>=24.1.0`
Type stubs for aiofiles.
Docs: https://github.com/python/typeshed

## Frontend Assets (CDN) #dependencies

### htmx `2.0.10`
Loaded from jsDelivr CDN with SRI in the base layout; infrastructure for dynamic browser actions (no partial endpoints yet).
Docs: https://htmx.org/
Used in: [src/news/templates/base.html:9](../src/news/templates/base.html#L9)

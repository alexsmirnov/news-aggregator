---
spec: init
scope: "init"
branch: init
repository: work
commit: "N/A (no commits yet)"
---

# Research: init

## Summary
The current repository state contains specification/task artifacts, a notebook-based workflow, and a Docker Compose runtime for Miniflux/PostgreSQL. No backend application scaffold files are present yet (no Python service source tree, no tests tree, no dependency manifests, no backend Dockerfile). The active specification is `init`, routed by `tasks/focus.txt`, targeting migration of `tasks/ai-news-digest.ipynb` behavior into an application codebase. Repository documentation includes FastAPI best-practices guidance, while HTMX appears only in external references (not in repository files).

## Detailed Findings

### Specification Routing and Task Artifacts
- Active specification is declared as `init` in `tasks/focus.txt:1`, with instructions pointing to `tasks/init.md` (`tasks/focus.txt:3`).
- Expected research and implementation-plan output paths are defined in `tasks/focus.txt:4-7`.
- The `init` PRD defines scaffold requirements (source layout, tests, dependency declaration, runtime/deployment scaffolding, root HTML response behavior) in `tasks/init.md:10-17` and success criteria in `tasks/init.md:24-28`.
- Current behavior baseline is explicitly documented as notebook-based workflow in `tasks/init.md:32`.

### Repository-Level Structure Relevant to `init`
- Top-level files currently present include `docker-compose.yml`, `.aiswe.yaml`, `README.md`, `CLAUDE.md`, and `tasks/*`.
- No Python source files (`*.py`) were found.
- No test files (`test_*.py`, `*_test.py`, `conftest.py`) were found.
- No dependency manifests (`pyproject.toml`, `requirements*.txt`, `Pipfile`, `poetry.lock`, `uv.lock`) were found.
- No Docker runtime definition for backend service (`Dockerfile*`) was found.
- `README.md` and `CLAUDE.md` are empty (`README.md`, `CLAUDE.md`).

### Existing Deployment Stack (Compose Context)
- `docker-compose.yml` defines two services only: `miniflux` and `db` (`docker-compose.yml:2`, `docker-compose.yml:22`).
- `miniflux` service uses image `miniflux/miniflux:latest` by default (`docker-compose.yml:3`) and maps host `4080` to container `8080` (`docker-compose.yml:6-7`).
- `miniflux` is configured with DB connection and bootstrap variables (`docker-compose.yml:12-18`), and depends on healthy `db` (`docker-compose.yml:8-10`).
- `db` service uses `postgres:latest` (`docker-compose.yml:23`) with database credentials and persistent volume mount (`docker-compose.yml:25-30`).
- Volume `miniflux-db` is declared external (`docker-compose.yml:35-37`).

### Project Technology Declaration
- `.aiswe.yaml` declares project language as Python (`.aiswe.yaml:1`).
- Declared technology tags are `langgraph`, `ai`, and `fastapi` (`.aiswe.yaml:2-5`).
- `rules/index.md` includes FastAPI ruleset linkage (`rules/index.md:6`).

### Notebook Workflow Mapped for Migration Context
- Environment bootstrap loads `.env.openai` using `python-dotenv` (`tasks/ai-news-digest.ipynb:10-13`).
- External clients are initialized for Miniflux and OpenAI-compatible routing (`tasks/ai-news-digest.ipynb:42-50`), with helper message constructors and `chat()` wrapper (`tasks/ai-news-digest.ipynb:51-60`).
- Model connectivity probing is performed across multiple model IDs and Responses API (`tasks/ai-news-digest.ipynb:62-71`).
- News ingestion flow:
  - Category lookup by title `news` via Miniflux (`tasks/ai-news-digest.ipynb:109-112`).
  - 24-hour window computation and category entry fetch (`tasks/ai-news-digest.ipynb:114-128`).
  - HTML-to-text extraction with BeautifulSoup (`tasks/ai-news-digest.ipynb:121-123`).
  - Data normalization into pandas DataFrame and entity-formatted text payload (`tasks/ai-news-digest.ipynb:130-159`).
- LLM processing flow:
  - Trending-news context query via model `sonar-reasoning-pro` (`tasks/ai-news-digest.ipynb:201-204`).
  - Digest-generation prompt and XML-like `<news>` output request via `gemini-flash` (`tasks/ai-news-digest.ipynb:223-274`).
  - `<news>` block extraction/parsing using regex + `xml.etree.ElementTree` fallback (`tasks/ai-news-digest.ipynb:302-342`).
  - Per-record refinement via `gemini-flash` and `url_context` tool parameter (`tasks/ai-news-digest.ipynb:354-375`).
- Output rendering flow:
  - Collapsible HTML rendering of refined records through IPython display (`tasks/ai-news-digest.ipynb:657-687`).

### Documentation Assets for FastAPI and Related Practices
- Repository has a FastAPI practices document with implementation examples and patterns (`rules/FASTAPI_BEST_PRACTICES.md:1-630`).
- The document covers lifespan/resource management (`rules/FASTAPI_BEST_PRACTICES.md:5-27`), service/repository/router separation (`rules/FASTAPI_BEST_PRACTICES.md:146-155`), router organization (`rules/FASTAPI_BEST_PRACTICES.md:257-341`), app assembly (`rules/FASTAPI_BEST_PRACTICES.md:343-383`), configuration patterns (`rules/FASTAPI_BEST_PRACTICES.md:385-443`), and testing examples (`rules/FASTAPI_BEST_PRACTICES.md:594-630`).
- No repository files referencing HTMX were found.

### Current Component Interaction Map
- `tasks/focus.txt` routes execution context to `tasks/init.md` (`tasks/focus.txt:1-3`).
- `tasks/init.md` references notebook as current non-application behavior (`tasks/init.md:32`) and constrains deployment target to compose-managed services (`tasks/init.md:34`).
- `docker-compose.yml` provides live infrastructure context consumed by notebook through Miniflux API variables (`docker-compose.yml:2-18` with notebook client init at `tasks/ai-news-digest.ipynb:42-45`).
- Notebook LLM interactions are routed through OpenAI-compatible client using environment-provided base URL and API key (`tasks/ai-news-digest.ipynb:50`, `tasks/ai-news-digest.ipynb:55-58`).

### FastAPI + HTMX Configuration Research (External, Descriptive)
- FastAPI+HTMX is documented externally as server-rendered HTML + partial HTML swaps over AJAX-like requests (`https://htmx.org/docs/`, `https://testdriven.io/blog/fastapi-htmx/`).
- Described advantages include reduced frontend build tooling, HTML-fragment interaction model, and direct server-side templating workflows (`https://htmx.org/essays/hypermedia-driven-applications/`, `https://blakecrosley.com/guides/fastapi-htmx`).
- Described disadvantages include reduced SPA-style component ecosystem, limited TypeScript-centric workflows, and server-driven rendering trade-offs (`https://blakecrosley.com/guides/fastapi-htmx`).
- Common alternatives for Python AI web applications are documented as:
  - FastAPI + Jinja templating (server-rendered pages)
  - FastAPI + React/Vue (API + SPA split)
  - Streamlit
  - Gradio
  - Django
  - Flask
  - NiceGUI
  (sources in External references section)

## Code References
- `tasks/focus.txt:1-7` - Active specification routing and expected documentation/plan artifact paths.
- `tasks/init.md:1-34` - Full PRD for scaffold initialization scope and constraints.
- `docker-compose.yml:1-37` - Existing compose stack (Miniflux + PostgreSQL) and runtime environment variables.
- `.aiswe.yaml:1-5` - Language and technology declarations.
- `rules/index.md:1-17` - Team rules index and FastAPI ruleset registration.
- `rules/FASTAPI_BEST_PRACTICES.md:1-630` - Repository FastAPI architecture/practice reference content.
- `tasks/ai-news-digest.ipynb:10-13` - Environment loading from `.env.openai`.
- `tasks/ai-news-digest.ipynb:42-71` - Miniflux/OpenAI client setup and multi-model probe calls.
- `tasks/ai-news-digest.ipynb:109-159` - Feed retrieval, content extraction, DataFrame build, and prompt payload formatting.
- `tasks/ai-news-digest.ipynb:201-205` - Trending-news generation call.
- `tasks/ai-news-digest.ipynb:223-275` - Digest synthesis prompt and generation call.
- `tasks/ai-news-digest.ipynb:302-342` - Regex/XML parsing pipeline for `<news>` blocks.
- `tasks/ai-news-digest.ipynb:354-375` - Source-link refinement function using tool-enabled LLM call.
- `tasks/ai-news-digest.ipynb:657-687` - HTML presentation for notebook output.

## External references
- https://fastapi.tiangolo.com/ - FastAPI framework documentation and benchmarks index.
- https://fastapi.tiangolo.com/benchmarks/ - FastAPI benchmark positioning.
- https://htmx.org/docs/ - HTMX request/response model and attributes.
- https://htmx.org/essays/hypermedia-driven-applications/ - Hypermedia-driven architecture description.
- https://testdriven.io/blog/fastapi-htmx/ - FastAPI + HTMX integration walkthrough.
- https://blakecrosley.com/guides/fastapi-htmx - FastAPI+HTMX operational comparison narrative.
- https://streamlit.io/ - Streamlit framework overview.
- https://gradio.app/ - Gradio framework overview.
- https://www.squadbase.dev/en/blog/streamlit-vs-gradio-in-2025-a-framework-comparison-for-ai-apps - Streamlit vs Gradio characteristics.
- https://www.techempower.com/benchmarks/ - Framework benchmark dataset (cross-framework context).

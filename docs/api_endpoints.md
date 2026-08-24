# API Endpoints

Digest page routes serve server-rendered HTML from the JSON digest archive; the `/aggregate` endpoints expose scheduler status and trigger immediate pipeline runs. Page routes are hidden from the OpenAPI schema (`include_in_schema=False`). #api #endpoint

## Digest Pages #endpoint

### GET, HEAD / #endpoint
Home page: renders the most recent digest of the first configured aggregation. Returns the HTML 404 page when the archive is unreadable.
Implementation: [src/news/pages.py:47-58](../src/news/pages.py#L47-L58)

### GET, HEAD /digest/{name} #endpoint
Latest digest for the aggregation `name` (e.g. `news`, `economy`, `technology`). Unknown aggregation names return the HTML 404 page.
Implementation: [src/news/pages.py:61-72](../src/news/pages.py#L61-L72)

### GET, HEAD /digest/{name}/{day} #endpoint
Digest for the aggregation `name` on the ISO date `day` (`YYYY-MM-DD`). Invalid or malformed dates and missing days return the HTML 404 page.
Implementation: [src/news/pages.py:75-91](../src/news/pages.py#L75-L91)

All page routes render through `_render_digest`, which builds a `DigestPage` view model via `archive.build_page` and renders [src/news/templates/digest.html](../src/news/templates/digest.html) ([src/news/pages.py:33-44](../src/news/pages.py#L33-L44)).

## Aggregate Control #endpoint

### GET /aggregate #endpoint
JSON status of the scheduled digest job: `job_id`, `scheduler_running`, `job_pending`, `next_run_time`. Returns 503 when the scheduler or job is unavailable.
Implementation: [src/news/aggregate.py:41-43](../src/news/aggregate.py#L41-L43), status builder [src/news/aggregate.py:18-38](../src/news/aggregate.py#L18-L38)

### POST /aggregate #endpoint
Triggers an immediate pipeline run by rescheduling the job to now; returns 202 with the updated status. Returns 503 when the scheduler is stopped or missing.
Implementation: [src/news/aggregate.py:46-63](../src/news/aggregate.py#L46-L63)

## Static Assets and Error Pages #endpoint

- `/style.css`, `/favicon.svg` and other files served from `src/news/frontend/` via the frontend mount with `404.html` fallback for undefined paths ([src/news/server.py:18](../src/news/server.py#L18))
- Unknown page routes return the static HTML 404 page ([src/news/frontend/404.html](../src/news/frontend/404.html), handler [src/news/pages.py:25-30](../src/news/pages.py#L25-L30))
- FastAPI built-in docs remain available at `/docs`, `/redoc`, `/openapi.json`

htmx 2.0.10 is loaded in the base layout with SRI ([src/news/templates/base.html:9](../src/news/templates/base.html#L9)); current endpoints are fully server-rendered, and no htmx partial endpoints exist yet.

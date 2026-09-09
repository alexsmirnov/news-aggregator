# Configuration & Environment

All configuration is defined in `Settings` (pydantic-settings `BaseSettings`, loads `.env`, process environment overrides take precedence) at [src/news/settings.py:49-78](../src/news/settings.py#L49-L78). #config #environment

## Required Variables #config #env

| Variable | Type | Purpose |
|---|---|---|
| `MINIFLUX_API_BASE` #env | URL | Miniflux API endpoint, e.g. `http://miniflux:8080` ([settings.py:49](../src/news/settings.py#L49)) |
| `MINIFLUX_API_KEY` #env | secret | Miniflux API token, sent as `X-Auth-Token` ([settings.py:50](../src/news/settings.py#L50), [miniflux_client.py:43](../src/news/digest/miniflux_client.py#L43)) |
| `LITELLM_API_KEY` #env | secret | API key for the OpenAI-compatible LLM router ([settings.py:51](../src/news/settings.py#L51)) |
| `LITELLM_ROUTER` #env | URL | LLM router base URL, e.g. `http://host.docker.internal:4000/v1` ([settings.py:52](../src/news/settings.py#L52)) |
| `DIGEST_OUTPUT_DIR` #env | path | Directory where digest JSON files are written, e.g. `/app/digests` ([settings.py:53](../src/news/settings.py#L53)) |

## Optional Variables (defaults) #config #env

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `AGGREGATIONS` #env | JSON list of `{name, miniflux_category, focus}` | `news`, `economy`, `technology` ([settings.py:39-43](../src/news/settings.py#L39-L43)) | Digest definitions: page name, Miniflux category title, LLM focus text ([settings.py:55](../src/news/settings.py#L55)) |
| `FETCH_LOOKBACK_HOURS` #env | int | `24` | Entry fetch window in hours ([settings.py:57](../src/news/settings.py#L57)) |
| `FETCH_LIMIT` #env | int | `10000` | Max entries fetched per aggregation ([settings.py:58](../src/news/settings.py#L58)) |
| `ENTRY_CONTENT_MAX_CHARS` #env | int | `20000` | Content truncation after HTML stripping ([settings.py:59](../src/news/settings.py#L59)) |
| `GROUPING_CONTENT_MAX_CHARS` #env | int | `300` | Per-entry content limit in the grouping prompt ([settings.py:60](../src/news/settings.py#L60)) |
| `GROUPING_K_SIGMA` #env | float | `3.0` | `MapReduceGrouping` clustering tightness: std devs above the corpus random-pair similarity baseline ([settings.py:64](../src/news/settings.py#L64)) |
| `GROUPING_MAP_CLUSTERS` #env | int | `40` | `MapReduceGrouping` recall knob: how many top-ranked clusters get a map (titling) LLM call ([settings.py:65](../src/news/settings.py#L65)) |
| `GROUPING_MAX_RECORDS` #env | int | `20` | `MapReduceGrouping` cost knob: max merged records returned to `refine_all` ([settings.py:66](../src/news/settings.py#L66)) |
| `REFINE_MAX_LINKS` #env | int | `10` | Max links passed to the refinement model per group ([settings.py:67](../src/news/settings.py#L67)) |
| `MODEL_TRENDING` #env | str | `sonar-reasoning-pro` | Trending model name (currently unused by the pipeline) ([settings.py:62](../src/news/settings.py#L62)) |
| `MODEL_GROUPING` #env | str | `gpt-5-terra` | Model for news grouping ([settings.py:63](../src/news/settings.py#L63)) |
| `MODEL_REFINEMENT` #env | str | `gemini-flash` | Model for summary refinement ([settings.py:64](../src/news/settings.py#L64)) |
| `EVAL_JUDGE_MODEL` #env | str | `gpt-5-luna` | LLM judge model used by the evaluation test suite ([settings.py:65](../src/news/settings.py#L65)) |
| `MODEL_EMBEDDING` #env | str | `bge-embed` | Model for the map-reduce grouping design's embedding step (not yet wired into the pipeline) ([settings.py:67](../src/news/settings.py#L67)) |
| `EMBEDDING_DIMENSIONS` #env | int | `1024` | Embedding vector dimensionality requested from the embedding model ([settings.py:68](../src/news/settings.py#L68)) |
| `RETRY_ATTEMPTS` #env | int | `3` | Declared retry attempt count ([settings.py:66](../src/news/settings.py#L66)) |
| `RETRY_MIN_WAIT_S` #env | int | `2` | Declared minimum retry wait ([settings.py:67](../src/news/settings.py#L67)) |
| `RETRY_MAX_WAIT_S` #env | int | `30` | Declared maximum retry wait ([settings.py:68](../src/news/settings.py#L68)) |
| `SCHEDULE_CRON` #env | crontab expression | `0 7,12,17 * * *` | Crontab expression (minute hour day month weekday) for scheduled pipeline runs — default 7AM, 12PM, and 5PM ([settings.py:77](../src/news/settings.py#L77)) |
| `SCHEDULE_TIMEZONE` #env | IANA timezone | `America/Los_Angeles` | Timezone the cron schedule is evaluated in ([settings.py:78](../src/news/settings.py#L78)) |

Note: the clients currently use their own constructor defaults for retry parameters, which match the settings defaults ([llm_client.py:31-33](../src/news/digest/llm_client.py#L31-L33), [miniflux_client.py:35-38](../src/news/digest/miniflux_client.py#L35-L38)).

## Topic Focus Texts #config

Default grouping focus injected into the LLM grouping prompt per aggregation: `NEWS_FOCUS`, `Economy_FOCUS`, `Technology_FOCUS` ([src/news/settings.py:14-37](../src/news/settings.py#L14-L37)).

## CLI Options #config

`news` CLI ([src/news/__init__.py:66-82](../src/news/__init__.py#L66-L82)):

- `news server [--host 0.0.0.0] [--port 4090] [--reload | --no-reload]` — run the web server (default command)
- `news aggregate` — run the full aggregation pipeline once without the server

See [Deployment](deployment_infrastructure.md) for how these variables are supplied in Docker Compose.

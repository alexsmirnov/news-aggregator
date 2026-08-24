# Deployment & Infrastructure

The application is deployed as a Docker container via Docker Compose, together with Miniflux (RSS feed aggregator) and PostgreSQL (Miniflux storage). The news application itself stores digests as JSON files on a volume; it has no database of its own. #deployment #infrastructure

## Docker Image #deployment

Multi-stage build defined in [Dockerfile](../Dockerfile):

- **Builder stage:** `python:3.14-slim` base with `uv` 0.12.5 copied from `ghcr.io/astral-sh/uv`; installs locked runtime dependencies with `uv sync --locked --no-dev` using a uv cache mount ([Dockerfile:3-20](../Dockerfile#L3-L20))
- **Runtime stage:** copies the built `.venv`, exposes port 4090, and runs `news server --host 0.0.0.0 --port 4090 --no-reload` ([Dockerfile:22-32](../Dockerfile#L22-L32))
- Build context restricted by [.dockerignore](../.dockerignore) to `Dockerfile`, `pyproject.toml`, `uv.lock`, `README.md`, and `src/`

## Compose Stack #deployment #infrastructure

[docker-compose.yml](../docker-compose.yml) defines three services:

| Service | Image | Ports | Role |
|---|---|---|---|
| `news` | built from `Dockerfile` | `4090:4090` | FastAPI digest application ([docker-compose.yml:2-23](../docker-compose.yml#L2-L23)) |
| `miniflux` | `${MINIFLUX_IMAGE:-miniflux/miniflux:latest}` | `4080:8080` | RSS feed aggregator, news source ([docker-compose.yml:24-43](../docker-compose.yml#L24-L43)) |
| `db` | `postgres:18` | not exposed | Miniflux database with `pg_isready` healthcheck ([docker-compose.yml:44-56](../docker-compose.yml#L44-L56)) |

Service wiring:

- `news` depends on `miniflux` and reaches it at `MINIFLUX_API_BASE=http://miniflux:8080` ([docker-compose.yml:10-15](../docker-compose.yml#L10-L15))
- `news` reaches the LLM router on the host via `extra_hosts` host-gateway mapping and `LITELLM_ROUTER=http://host.docker.internal:4000/v1` ([docker-compose.yml:12-18](../docker-compose.yml#L12-L18))
- `miniflux` waits for the `db` healthcheck, runs its own migrations (`RUN_MIGRATIONS=1`), and bootstraps an admin user ([docker-compose.yml:30-40](../docker-compose.yml#L30-L40))
- Secrets (`MINIFLUX_API_KEY`, `LITELLM_API_KEY`) are loaded from the host `.env` via `env_file` ([docker-compose.yml:20-21](../docker-compose.yml#L20-L21))

## Storage #infrastructure

- Digest JSON files are written to `DIGEST_OUTPUT_DIR=/app/digests` inside the container, persisted on the `news-digests` volume ([docker-compose.yml:19-23](../docker-compose.yml#L19-L23))
- Miniflux data lives on the `miniflux-db` volume mounted at `/var/lib/postgresql` ([docker-compose.yml:51-52](../docker-compose.yml#L51-L52))
- Both volumes are declared `external: true` and must exist before the stack starts ([docker-compose.yml:57-61](../docker-compose.yml#L57-L61))

## External Dependencies #infrastructure

- **LiteLLM router** (OpenAI-compatible API, expected at `host.docker.internal:4000/v1`): not part of the Compose stack; provided by the host environment
- **Miniflux categories** named per the configured aggregations (`news`, `Economy`, `Technology`) must exist in the Miniflux instance, since the pipeline resolves categories by title ([src/news/digest/miniflux_client.py:57-96](../src/news/digest/miniflux_client.py#L57-L96))

See [Configuration](config_environment.md) for the full environment variable reference and [Architecture Overview](architecture_overview.md) for the pipeline design.

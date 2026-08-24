# Breaking News Aggregator

Web application that analyzes RSS news sources collected by a self-hosted Miniflux feed aggregator, uses LLM calls to group trending news and generate summaries, and serves the resulting digests as plain HTML pages. Built with Python 3.14 and FastAPI; the aggregation pipeline runs as a periodical APScheduler job inside the server process; AI calls go through an OpenAI-compatible LiteLLM router; pages are rendered with Jinja2 templates, plain CSS, and htmx. Deployed as a Docker container via Docker Compose together with Miniflux and PostgreSQL. #index #overview #python #fastapi #llm

## Documentation

- [Architecture Overview](architecture_overview.md) - System design, digest pipeline data flow, and technology stack
- [API Endpoints](api_endpoints.md) - Digest page routes and aggregate control endpoints
- [Packages & Modules](packages_modules.md) - Internal package structure and responsibilities
- [Deployment](deployment_infrastructure.md) - Docker image, Compose stack, and infrastructure
- [Configuration](config_environment.md) - Environment variables and application settings
- [Tests](tests_coverage.md) - Test files and coverage
- [Aggregator Evaluation](evaluation_aggregator.md) - LLM grouping/summary quality evaluation: datasets, metrics, thresholds
- [Dependencies](dependencies_libraries.md) - External libraries and tools

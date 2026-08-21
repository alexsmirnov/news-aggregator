# Breaking news aggregator
## Project goal
Web application to analyze different news sources and generate digest.
User can read those digests in web interface.
News sources:
- RSS feeds collected by Miniflux feed aggregator
Aggregation:
- LLM call to group most trending news
- Direct read feed links for more detailed information
- LLM call to generate summaries

# Technology

FastAPI Python application, run in docker container.

Aggregator runs as periodical job inside fastapi server.
AI calls using OpenAI compatible router and client library
Uv tool to manage dependencies
Plain HTML generated from jinja2 templates and aggregated news
Plain CSS
Dynamic content and actions in browser use htmx

# Deployment

Application deployed as docker container by docker compose, together with Miniflux and postgresql
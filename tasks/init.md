# Initialize Backend Project Scaffold for Breaking-News Web Application

## What
Create the initial Python backend project scaffold for a web application that will migrate the notebook-based breaking-news workflow into an application codebase. The scaffold must define baseline project artifacts for source code, tests, dependency management, execution environment, and deployment scaffolding, and include a minimal HTML response endpoint at the root path.

## Why
This is required to unblock migration from `tasks/ai-news-digest.ipynb` to a complete, maintainable application. A standardized scaffold enables implementation work to proceed with clear technical boundaries and runnable deployment context.

## Scope
### In Scope:
- Initial backend project structure for Python service development
- Baseline test structure for backend code
- Dependency declaration artifacts
- Execution environment definition for local/container runtime
- Deployment scaffolding compatible with existing `docker compose` workflow
- Runtime behavior requirement: root path (`/`) serves a "Hello Word" HTML page after deployment

### Out of Scope:
- Breaking-news aggregation logic and content-processing features
- Changes to external API assumptions and contracts
- Feature-level data ingestion, clustering, summarization, or ranking behavior
- UI features beyond minimal root-path HTML response requirement

## Success Criteria
- [ ] Project scaffold artifacts are present for source, tests, dependencies, and execution environment
- [ ] Deployment scaffolding supports running the application via `docker compose` in the current stack context
- [ ] After deployment with `docker compose`, the application is running and `GET /` returns a "Hello Word" HTML page
- [ ] The brief preserves external API assumptions without modification

## Context (if applicable)
- Affected area: backend service initialization and deployment scaffolding
- Current behavior: notebook-based workflow in `tasks/ai-news-digest.ipynb` without application scaffold
- Constraint: external API assumptions must remain unchanged
- Constraint: target runtime is Docker container deployment alongside existing compose-managed services

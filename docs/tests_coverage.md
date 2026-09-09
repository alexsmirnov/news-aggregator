# Tests & Coverage

Pytest suite in `tests/` with `asyncio_mode = "auto"`, module-scoped event loop, and an `integration` marker for tests with real network I/O excluded from unit runs ([pyproject.toml:49-67](../pyproject.toml#L49-L67)). Shared fixtures (`TestClient` with temp settings, digest seeding, HTML text helpers) live in [tests/conftest.py](../tests/conftest.py). #tests #testing

## Unit Test Files

- [test_init.py](../tests/test_init.py) - CLI: argument parsing defaults/overrides/errors, logging configuration, Uvicorn log config, `main()` dispatch to server vs aggregate
- [test_settings.py](../tests/test_settings.py) - Settings loading from environment and `.env`, required-field validation, environment-over-dotenv precedence
- [test_server.py](../tests/test_server.py) - `run_aggregate` wiring; `GET/POST /aggregate` status, reschedule, and 503 paths; home page rendering, static assets, HTML 404, OpenAPI visibility, `/docs` endpoints
- [test_scheduler.py](../tests/test_scheduler.py) - App creation without credentials, lifespan validation errors, cron job options (`max_instances`, `coalesce`), scheduler start/shutdown lifecycle
- [test_miniflux_client.py](../tests/test_miniflux_client.py) - Category lookup and auth header, entry mapping, pagination, response validation, retry/give-up on transient HTTP errors, factory context manager (uses `httpx.MockTransport`)
- [test_llm_client.py](../tests/test_llm_client.py) - `chat` and `chat_parsed` results, retry on rate limits, reraise after exhaustion, non-transient errors not retried, factory cleanup
- [test_service.py](../tests/test_service.py) - Largest suite: HTML stripping, fetch window/limit, refinement link matching and failure isolation, concurrency order, digest writing, full pipeline happy/partial/empty/failure paths (grouping injected as a `FakeGrouping`), multi-aggregation sequencing
- [test_grouping.py](../tests/test_grouping.py) - Entry formatting/truncation for the grouping prompt, `LlmGrouping` call shape (model, messages, response format), `PipelineError` on empty grouping response
- [test_archive.py](../tests/test_archive.py) - Digest path layout, available-date discovery (ordering, malformed files, prefix collisions), digest loading, `build_page` navigation, edge cases, link scheme filtering
- [test_pages.py](../tests/test_pages.py) - HTTP-level rendering: aggregation/date selection, 404 paths, empty and corrupt archives, `<details>` rendering, XSS escaping, `javascript:` link filtering, navigation links, HEAD support, redirects
- [test_prompts.py](../tests/test_prompts.py) - Exact prompt text assertions: focus embedding, date grounding, required summary sections, link joining
- [test_map_reduce.py](../tests/test_map_reduce.py) - `MapReduceGrouping`: embedding batching/normalization/truncation/fallback, `calibrate_threshold` corpus-similarity sampling and clamping, `cluster` average-linkage behavior (singleton survival, chaining resistance), scoring by hostname entropy/arrival bursts/novelty, positive singleton scores, stable ranking and boundary inputs; `__call__` end-to-end clustering/mapping/reduction into `NewsRecord`s, map-call count capped at `grouping_map_clusters`, merge-by-index link unioning without duplicates, unreferenced candidates surviving a merge pass, out-of-range member indexes ignored, map-call failure falling back to the cluster's entry title, output ranked by `trend_score` and truncated to `grouping_max_records`, collapse loop termination when a reduce pass does not shrink the level, `PipelineError` on an empty reduce response

## Evaluation Suite (tests/evaluation/) #tests

Frozen-dataset LLM quality evaluation; requires LLM credentials and is skipped otherwise. See [Aggregator Evaluation](evaluation_aggregator.md) for the detailed reference (datasets, metrics, thresholds, judge setup).

- [conftest.py](../tests/evaluation/conftest.py) - Dataset discovery from `data/rss_entries_*.json`, grouping implementation registry (`GROUPING_IMPLEMENTATIONS`), judge model setup (deepeval `GPTModel` via the LiteLLM router), package-scoped real grouping/refinement runs shared across both eval test files
- [test_grouping_eval.py](../tests/evaluation/test_grouping_eval.py) - Threshold-gated grouping metrics per dataset x grouping implementation: pairwise F1 >= 0.6, GEval grouping correctness >= 0.6
- [test_summary_eval.py](../tests/evaluation/test_summary_eval.py) - Threshold-gated summary metrics (default grouping implementation only): ROUGE-L mean >= 0.3, GEval summary faithfulness >= 0.6
- [test_cluster_eval.py](../tests/evaluation/test_cluster_eval.py) - Not quality-threshold-gated: embeds each dataset's frozen entries, calibrates/clusters content vectors, and scores the final sigma=3.2 result. Writes score-ranked groups with titles, links, and scoring metadata to repository-relative `tmp/cluster_eval_<dataset_id>.yaml`; verifies YAML metadata, entry conservation, group sizes, and ordering without requiring gold groups
- [test_metrics.py](../tests/evaluation/test_metrics.py) - Deterministic metric implementations: pairwise precision/recall/F1, ROUGE-L, greedy group matching, link-set extraction
- [metrics.py](../tests/evaluation/metrics.py) - Metric helpers: pair-based clustering metrics, ROUGE-L scoring, Jaccard group matching, link-set extraction
- [capture_dataset.py](../tests/evaluation/capture_dataset.py) - Standalone utility that freezes a Miniflux RSS snapshot into `data/rss_entries_<date>_<category>.json`
- [test_capture_dataset.py](../tests/evaluation/test_capture_dataset.py) - Capture script behavior with mocked service and snapshot writing

Frozen fixtures in [tests/evaluation/data/](../tests/evaluation/data/): RSS snapshots for two dates/categories, plus human-verified expected groups and summaries for `2026_07_22_Economy`.

## Development Environments

- `testenv/` - gitignored development/test virtualenv (Python 3.14), excluded from Pyright analysis ([pyproject.toml:94-99](../pyproject.toml#L94-L99))
- `.deepeval/` - gitignored telemetry state created by the deepeval library during evaluation runs

See [Dependencies](dependencies_libraries.md) for the testing/evaluation libraries used.

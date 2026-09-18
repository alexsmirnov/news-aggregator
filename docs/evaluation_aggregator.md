# Aggregator Evaluation

Quality evaluation for the LLM stages of the digest pipeline — news grouping and summary refinement — implemented as an integration-marked pytest suite in [tests/evaluation/](../tests/evaluation/). It runs the real `DigestService` LLM calls against frozen RSS datasets and scores the output with deterministic clustering/text metrics and LLM-as-a-judge metrics, gated by quality thresholds. #evaluation #testing #llm

## Evaluation Targets #evaluation

The suite evaluates the two LLM stages of the pipeline (see [Architecture Overview](architecture_overview.md)):

1. **Grouping** — does a `Grouping` implementation ([src/news/digest/grouping.py](../src/news/digest/grouping.py)) cluster entries about the same real-world event together? Two implementations are registered: `LlmGrouping` (default) and the cluster-then-summarize `MapReduceGrouping` ([src/news/digest/map_reduce.py](../src/news/digest/map_reduce.py)).
2. **Summarization** — does `DigestService.refine_all` ([src/news/digest/service.py:201-233](../src/news/digest/service.py#L201-L233)) produce summaries that cover the key facts of human-written reference summaries without invented facts?

Both stages are executed for real against the configured LLM router (`LITELLM_ROUTER` / `LITELLM_API_KEY`); only the Miniflux client is stubbed out with a dummy object since entries come from frozen files ([tests/evaluation/conftest.py:151-156](../tests/evaluation/conftest.py#L151-L156)).

## Grouping Implementation Registry #evaluation

`GROUPING_IMPLEMENTATIONS` ([conftest.py:33-38](../tests/evaluation/conftest.py#L33-L38)) maps a name to a `(Settings, LlmClient) -> Grouping` factory; it currently holds two entries, `"llm"` (`LlmGrouping`) and `"map_reduce"` (`MapReduceGrouping`). The suite is parametrized over this registry via the `grouping_name` fixture ([conftest.py:69-71](../tests/evaluation/conftest.py#L69-L71)), so every grouping test id carries the implementation name (e.g. `test_grouping_pairwise_f1[2026_07_22_Economy-llm]`, `test_grouping_pairwise_f1[2026_07_22_Economy-map_reduce]`). Adding a new implementation to compare means adding one entry here — no test changes required. `DEFAULT_GROUPING = "llm"` ([conftest.py:39](../tests/evaluation/conftest.py#L39)) is the implementation summary evaluation runs against (see below); switching the production default in `src/news/server.py` is a separate decision made on these evaluation numbers.

## Quality Thresholds #evaluation

Constants in [tests/evaluation/conftest.py:28-30](../tests/evaluation/conftest.py#L28-L30):

| Constant | Value | Applies to |
|---|---|---|
| `GROUPING_F1_MIN` | `0.6` | Pairwise F1 of grouping link clusters |
| `ROUGE_L_MIN` | `0.3` | Mean ROUGE-L F-measure of matched summaries |
| `JUDGE_MIN` | `0.6` | GEval judge score (grouping correctness and summary faithfulness) |

## Evaluation Tests #evaluation

The suite is split into two files, both marked `@pytest.mark.integration` (real LLM I/O, excluded from unit runs per the marker definition in [pyproject.toml:65-67](../pyproject.toml#L65-L67)) and parametrized over every discovered dataset **and** every registered grouping implementation:

| Test | Metric | Assertion |
|---|---|---|
| `test_grouping_pairwise_f1` ([test_grouping_eval.py:14-32](../tests/evaluation/test_grouping_eval.py#L14-L32)) | `pairwise_prf` over predicted vs expected link sets | F1 >= 0.6 |
| `test_grouping_judge` ([test_grouping_eval.py:35-67](../tests/evaluation/test_grouping_eval.py#L35-L67)) | deepeval `GEval` "Grouping correctness" over formatted entries (input), actual grouping JSON, expected grouping JSON | judge score >= 0.6 via `assert_test` |

`test_grouping_eval.py` therefore runs once per `(dataset, grouping implementation)` pair — it is the file to extend when comparing new grouping strategies.

| Test | Metric | Assertion |
|---|---|---|
| `test_summary_rouge_l` ([test_summary_eval.py:16-37](../tests/evaluation/test_summary_eval.py#L16-L37)) | `rouge_l` per matched group | mean >= 0.3 |
| `test_summary_judge_mean` ([test_summary_eval.py:41-72](../tests/evaluation/test_summary_eval.py#L41-L72)) | deepeval `GEval` "Summary faithfulness" per matched group, awaited sequentially | mean >= 0.6 |

`test_summary_eval.py` is collected for every `(dataset, grouping implementation)` combination too, but the underlying `refined_run` fixture skips (does not fail) whenever `grouping_name != DEFAULT_GROUPING` ([conftest.py:140-143](../tests/evaluation/conftest.py#L140-L143)) — refinement is expensive and its quality does not depend on which grouping implementation produced the input, so it only runs once per dataset against the default implementation.

Summary tests match predicted groups to expected groups first (`_matched_summaries`, [test_summary_eval.py:75-100](../tests/evaluation/test_summary_eval.py#L75-L100)): `match_groups` pairs groups by link overlap, then the expected group's title looks up the reference summary; groups without an expected summary are skipped with a warning.

## Manual Cluster Inspection #evaluation

[test_cluster_eval.py](../tests/evaluation/test_cluster_eval.py) is not quality-threshold-gated; it produces a human-readable artifact. `test_cluster_eval_writes_inspectable_yaml` is parametrized over every discovered `dataset_id`, requires no expected groups, and directly exercises `MapReduceGrouping` without registering it in `GROUPING_IMPLEMENTATIONS`. It embeds `frozen_entries`, clusters content vectors across sigma values 2.7 through 3.2, and scores only the final sigma=3.2 result. One YAML file per dataset is written to repository-relative `tmp/cluster_eval_<dataset_id>.yaml`, creating the directory as needed. Each group retains `entries: [{title, link}, ...]` and adds `size`, `unique_domains`, `source_entropy`, `effective_sources`, `burst_z`, `novelty`, and `trend_score`, sorted by descending trend score with singletons preserved. Diversity counts article hostnames; burst uses the current corpus rate baseline; novelty defaults to 1.0 because this evaluation supplies no historical embeddings. Assertions check score metadata, entry conservation, group sizes, and ordering after loading the YAML. The artifact path is logged for manual inspection.

## Deterministic Metrics #evaluation

Implemented in [tests/evaluation/metrics.py](../tests/evaluation/metrics.py) and unit-tested in [tests/evaluation/test_metrics.py](../tests/evaluation/test_metrics.py):

- **`pairwise_prf`** ([metrics.py:6-21](../tests/evaluation/metrics.py#L6-L21)) — converts each group's link set into all link pairs (`_cluster_pairs`), then computes precision/recall/F1 over the pair sets: a pair of links placed together in both predicted and expected groupings counts as a true positive. Order- and title-insensitive; judges only which items are co-clustered.
- **`rouge_l`** ([metrics.py:24-26](../tests/evaluation/metrics.py#L24-L26)) — ROUGE-L F-measure via `rouge-score` with stemming, reference vs candidate summary.
- **`match_groups`** ([metrics.py:29-57](../tests/evaluation/metrics.py#L29-L57)) — greedy one-to-one matching between predicted and expected groups: all pairs with non-empty link intersection are scored by Jaccard similarity (`_jaccard`, [metrics.py:77-78](../tests/evaluation/metrics.py#L77-L78)) and assigned in descending score order.
- **`links`** ([metrics.py:60-67](../tests/evaluation/metrics.py#L60-L67)) — extracts and validates a group's link set as `set[str]`, raising `ValueError` if `links` is missing or not all strings; shared by both grouping and summary tests.

## LLM Judge #evaluation #llm

Judge-based metrics use deepeval `GEval` with a `GPTModel` pointed at the same LiteLLM router as the pipeline ([conftest.py:94-100](../tests/evaluation/conftest.py#L94-L100)): model name from `EVAL_JUDGE_MODEL` (default `gpt-5-luna`, [src/news/settings.py:65](../src/news/settings.py#L65)), temperature 1.

Judge criteria ([test_grouping_eval.py:43-59](../tests/evaluation/test_grouping_eval.py#L43-L59), [test_summary_eval.py:103-116](../tests/evaluation/test_summary_eval.py#L103-L116)):

- **Grouping correctness** — items about the same real-world event must be in one group, different events must not be merged; wording is not judged. Uses input, actual output, and expected output.
- **Summary faithfulness** — the actual summary must faithfully cover the key facts of the expected summary without invented facts. Uses actual and expected output only.

## Frozen Datasets #evaluation

Datasets live in [tests/evaluation/data/](../tests/evaluation/data/) and are discovered at collection time by the filename pattern `rss_entries_<id>.json` ([conftest.py:40-51](../tests/evaluation/conftest.py#L40-L51)); each discovered id parametrizes the whole suite via the `dataset_id` fixture ([conftest.py:62-64](../tests/evaluation/conftest.py#L62-L64)).

| File | Contents |
|---|---|
| `rss_entries_<date>_<category>.json` | Frozen `RssEntry` lists captured from Miniflux (id, title, link, content, published_at, source — schema at [src/news/digest/schemas.py:6-12](../src/news/digest/schemas.py#L6-L12)) |
| `expected_groups_<id>.yaml` | Human-verified groups: `{title, entries: [{title, link}]}` per trending event |
| `expected_summaries_<id>.json` | Human-written reference summaries keyed by the expected group title: `{title, summary}` |

Currently available datasets:

| Dataset id | Entries | Expected groups/summaries |
|---|---|---|
| `2026_06_15_Economy` | 692 | none (summary tests skip) |
| `2026_06_15_news` | 1090 | none (summary tests skip) |
| `2026_07_22_Economy` | 1156 | 20 groups, 20 summaries |
| `2026_07_22_news` | 1362 | none (summary tests skip) |

Only `2026_07_22_Economy` has expected data, so grouping tests run for all discovered datasets while summary-dependent tests skip elsewhere via `_load_required_data` ([conftest.py:169-173](../tests/evaluation/conftest.py#L169-L173)).

## Dataset Capture Utility #evaluation

[capture_dataset.py](../tests/evaluation/capture_dataset.py) freezes a new RSS snapshot: it builds a real `MinifluxClient` and calls `DigestService.fetch_entries` for a category and date, then writes `data/rss_entries_<YYYY_MM_DD>_<category>.json` ([capture_dataset.py:21-41](../tests/evaluation/capture_dataset.py#L21-L41), [capture_dataset.py:61-80](../tests/evaluation/capture_dataset.py#L61-L80)). CLI options: `--category` (`news` | `Economy` | `Technology`, default `news`) and `--date` (`YYYY-MM-DD`, default today) ([capture_dataset.py:44-58](../tests/evaluation/capture_dataset.py#L44-L58)). The script is unit-tested with a mocked service in [tests/evaluation/test_capture_dataset.py](../tests/evaluation/test_capture_dataset.py). Expected groups/summaries for a new dataset must be authored by hand following the shapes above (groups as YAML, summaries as JSON).

## Fixtures and Execution Model #evaluation

**Package-scoped** fixtures in [tests/evaluation/conftest.py](../tests/evaluation/conftest.py) (`scope="package"`, async ones also `loop_scope="package"`) minimize LLM calls — one grouping run per `(dataset, grouping implementation)` pair and one refinement run per dataset, shared across both `test_grouping_eval.py` and `test_summary_eval.py`:

- `eval_settings` — loads `Settings`; skips the suite when credentials are not configured ([conftest.py:54-59](../tests/evaluation/conftest.py#L54-L59))
- `dataset_id` — parametrized over every discovered dataset id ([conftest.py:62-64](../tests/evaluation/conftest.py#L62-L64))
- `grouping_name` — parametrized over `GROUPING_IMPLEMENTATIONS`, sorted by name ([conftest.py:67-69](../tests/evaluation/conftest.py#L67-L69))
- `frozen_entries` — validates the dataset JSON into `RssEntry` objects ([conftest.py:72-80](../tests/evaluation/conftest.py#L72-L80))
- `judge` — deepeval `GPTModel` over the LiteLLM router ([conftest.py:93-100](../tests/evaluation/conftest.py#L93-L100))
- `grouping_run` — builds the `grouping_name` implementation from the registry and calls it on entries formatted with `grouping_content_max_chars`; returns the formatted prompt, the actual grouping JSON, and the `NewsRecord` list ([conftest.py:103-130](../tests/evaluation/conftest.py#L103-L130))
- `refined_run` — real `refine_all` call over the grouping results, dated today; skips unless `grouping_name == DEFAULT_GROUPING` so refinement cost does not multiply with the number of registered grouping implementations ([conftest.py:133-162](../tests/evaluation/conftest.py#L133-L162))

Package scope (rather than module scope) lets `grouping_run` be shared between the two split test files without re-running the grouping call per file. Splitting `tests/` from `tests/evaluation/` still works without an `__init__.py` in either directory (verified: `uv run pytest tests/evaluation --collect-only -q` resolves the parametrized, package-scoped async fixtures cleanly).

Behavior notes:

- Grouping always uses the focus of the first configured aggregation (`aggregations[0].focus`) regardless of the dataset's category ([conftest.py:121-124](../tests/evaluation/conftest.py#L121-L124)).
- Refinement in evaluation goes through the same `refine_record` path as production, including the `url_context` tool, so the judge scores summaries produced with live link reading ([src/news/digest/service.py:158-163](../src/news/digest/service.py#L158-L163)).
- Runs are non-deterministic by nature (LLM output); thresholds are set to absorb run-to-run variance, and the pairwise/ROUGE metrics provide the deterministic regression signal.

Related docs: [Tests & Coverage](tests_coverage.md) for the unit suites, [Configuration](config_environment.md) for `EVAL_JUDGE_MODEL` and credentials, [Dependencies](dependencies_libraries.md) for deepeval and rouge-score.

# Aggregator Evaluation

Quality evaluation for the LLM stages of the digest pipeline — news grouping and summary refinement — implemented as an integration-marked pytest suite in [tests/evaluation/](../tests/evaluation/). It runs the real `DigestService` LLM calls against frozen RSS datasets and scores the output with deterministic clustering/text metrics and LLM-as-a-judge metrics, gated by quality thresholds. #evaluation #testing #llm

## Evaluation Targets #evaluation

The suite evaluates the two LLM stages of the pipeline (see [Architecture Overview](architecture_overview.md)):

1. **Grouping** — does `extract_groups` ([src/news/digest/service.py:128-165](../src/news/digest/service.py#L128-L165)) cluster entries about the same real-world event together?
2. **Summarization** — does `refine_all` ([src/news/digest/service.py:264-295](../src/news/digest/service.py#L264-L295)) produce summaries that cover the key facts of human-written reference summaries without invented facts?

Both stages are executed for real against the configured LLM router (`LITELLM_ROUTER` / `LITELLM_API_KEY`); only the Miniflux client is stubbed out with a dummy object since entries come from frozen files ([tests/evaluation/conftest.py:96-100](../tests/evaluation/conftest.py#L96-L100)).

## Quality Thresholds #evaluation

Constants in [tests/evaluation/test_digest_eval.py:13-15](../tests/evaluation/test_digest_eval.py#L13-L15):

| Constant | Value | Applies to |
|---|---|---|
| `GROUPING_F1_MIN` | `0.6` | Pairwise F1 of grouping link clusters |
| `ROUGE_L_MIN` | `0.3` | Mean ROUGE-L F-measure of matched summaries |
| `JUDGE_MIN` | `0.6` | GEval judge score (grouping correctness and summary faithfulness) |

## Evaluation Tests #evaluation

All four tests are marked `@pytest.mark.integration` (real LLM I/O, excluded from unit runs per the marker definition in [pyproject.toml:65-67](../pyproject.toml#L65-L67)) and are parametrized over every discovered dataset:

| Test | Metric | Assertion |
|---|---|---|
| `test_grouping_pairwise_f1` ([test_digest_eval.py:19-37](../tests/evaluation/test_digest_eval.py#L19-L37)) | `pairwise_prf` over predicted vs expected link sets | F1 >= 0.6 |
| `test_grouping_judge` ([test_digest_eval.py:40-72](../tests/evaluation/test_digest_eval.py#L40-L72)) | deepeval `GEval` "Grouping correctness" over formatted entries (input), actual grouping JSON, expected grouping JSON | judge score >= 0.6 via `assert_test` |
| `test_summary_rouge_l` ([test_digest_eval.py:75-97](../tests/evaluation/test_digest_eval.py#L75-L97)) | `rouge_l` per matched group | mean >= 0.3 |
| `test_summary_judge_mean` ([test_digest_eval.py:100-132](../tests/evaluation/test_digest_eval.py#L100-L132)) | deepeval `GEval` "Summary faithfulness" per matched group, awaited sequentially | mean >= 0.6 |

Summary tests match predicted groups to expected groups first (`_matched_summaries`, [test_digest_eval.py:144-169](../tests/evaluation/test_digest_eval.py#L144-L169)): `match_groups` pairs groups by link overlap, then the expected group's title looks up the reference summary; groups without an expected summary are skipped with a warning.

## Deterministic Metrics #evaluation

Implemented in [tests/evaluation/metrics.py](../tests/evaluation/metrics.py) and unit-tested in [tests/evaluation/test_metrics.py](../tests/evaluation/test_metrics.py):

- **`pairwise_prf`** ([metrics.py:6-21](../tests/evaluation/metrics.py#L6-L21)) — converts each group's link set into all link pairs (`_cluster_pairs`), then computes precision/recall/F1 over the pair sets: a pair of links placed together in both predicted and expected groupings counts as a true positive. Order- and title-insensitive; judges only which items are co-clustered.
- **`rouge_l`** ([metrics.py:24-26](../tests/evaluation/metrics.py#L24-L26)) — ROUGE-L F-measure via `rouge-score` with stemming, reference vs candidate summary.
- **`match_groups`** ([metrics.py:29-57](../tests/evaluation/metrics.py#L29-L57)) — greedy one-to-one matching between predicted and expected groups: all pairs with non-empty link intersection are scored by Jaccard similarity (`_jaccard`, [metrics.py:68-69](../tests/evaluation/metrics.py#L68-L69)) and assigned in descending score order.

## LLM Judge #evaluation #llm

Judge-based metrics use deepeval `GEval` with a `GPTModel` pointed at the same LiteLLM router as the pipeline ([conftest.py:74-81](../tests/evaluation/conftest.py#L74-L81)): model name from `EVAL_JUDGE_MODEL` (default `gpt-5-luna`, [src/news/settings.py:65](../src/news/settings.py#L65)), temperature 1.

Judge criteria ([test_digest_eval.py:48-64](../tests/evaluation/test_digest_eval.py#L48-L64), [test_digest_eval.py:172-185](../tests/evaluation/test_digest_eval.py#L172-L185)):

- **Grouping correctness** — items about the same real-world event must be in one group, different events must not be merged; wording is not judged. Uses input, actual output, and expected output.
- **Summary faithfulness** — the actual summary must faithfully cover the key facts of the expected summary without invented facts. Uses actual and expected output only.

## Frozen Datasets #evaluation

Datasets live in [tests/evaluation/data/](../tests/evaluation/data/) and are discovered at collection time by the filename pattern `rss_entries_<id>.json` ([conftest.py:20-37](../tests/evaluation/conftest.py#L20-L37)); each discovered id parametrizes the whole suite via the `dataset_id` fixture ([conftest.py:48-50](../tests/evaluation/conftest.py#L48-L50)).

| File | Contents |
|---|---|
| `rss_entries_<date>_<category>.json` | Frozen `RssEntry` lists captured from Miniflux (id, title, link, content, published_at, source — schema at [src/news/digest/schemas.py:6-12](../src/news/digest/schemas.py#L6-L12)) |
| `expected_groups_<id>.json` | Human-verified groups: `{title, links[]}` per trending event |
| `expected_summaries_<id>.json` | Human-written reference summaries keyed by the expected group title: `{title, summary}` |

Currently available datasets:

| Dataset id | Entries | Expected groups/summaries |
|---|---|---|
| `2026_06_15_Economy` | 692 | none (summary tests skip) |
| `2026_06_15_news` | 1090 | none (summary tests skip) |
| `2026_07_22_Economy` | 1156 | 20 groups, 20 summaries |
| `2026_07_22_news` | 1362 | none (summary tests skip) |

Only `2026_07_22_Economy` has expected data, so grouping tests run for all four datasets while summary-dependent tests skip elsewhere via `_load_required_data` ([conftest.py:141-145](../tests/evaluation/conftest.py#L141-L145)).

## Dataset Capture Utility #evaluation

[capture_dataset.py](../tests/evaluation/capture_dataset.py) freezes a new RSS snapshot: it builds a real `MinifluxClient` and calls `DigestService.fetch_entries` for a category and date, then writes `data/rss_entries_<YYYY_MM_DD>_<category>.json` ([capture_dataset.py:21-41](../tests/evaluation/capture_dataset.py#L21-L41), [capture_dataset.py:61-80](../tests/evaluation/capture_dataset.py#L61-L80)). CLI options: `--category` (`news` | `Economy` | `Technology`, default `news`) and `--date` (`YYYY-MM-DD`, default today) ([capture_dataset.py:44-58](../tests/evaluation/capture_dataset.py#L44-L58)). The script is unit-tested with a mocked service in [tests/evaluation/test_capture_dataset.py](../tests/evaluation/test_capture_dataset.py). Expected groups/summaries for a new dataset must be authored by hand following the JSON shapes above.

## Fixtures and Execution Model #evaluation

Module-scoped fixtures in [tests/evaluation/conftest.py](../tests/evaluation/conftest.py) minimize LLM calls — one grouping run and one refinement run per dataset, shared by all four tests:

- `eval_settings` — loads `Settings`; skips the suite when credentials are not configured ([conftest.py:40-45](../tests/evaluation/conftest.py#L40-L45))
- `frozen_entries` — validates the dataset JSON into `RssEntry` objects ([conftest.py:53-61](../tests/evaluation/conftest.py#L53-L61))
- `judge` — deepeval `GPTModel` over the LiteLLM router ([conftest.py:74-81](../tests/evaluation/conftest.py#L74-L81))
- `grouping_run` — real `extract_groups` call on entries formatted with `grouping_content_max_chars`; returns the formatted prompt, the actual grouping JSON, and the `NewsRecord` list ([conftest.py:84-111](../tests/evaluation/conftest.py#L84-L111))
- `refined_run` — real `refine_all` call over the grouping results, dated today ([conftest.py:114-134](../tests/evaluation/conftest.py#L114-L134))

Behavior notes:

- Grouping always uses the focus of the first configured aggregation (`aggregations[0].focus`) regardless of the dataset's category ([conftest.py:102-105](../tests/evaluation/conftest.py#L102-L105)).
- Refinement in evaluation goes through the same `refine_record` path as production, including the `url_context` tool, so the judge scores summaries produced with live link reading ([src/news/digest/service.py:221-227](../src/news/digest/service.py#L221-L227)).
- Runs are non-deterministic by nature (LLM output); thresholds are set to absorb run-to-run variance, and the pairwise/ROUGE metrics provide the deterministic regression signal.

Related docs: [Tests & Coverage](tests_coverage.md) for the unit suites, [Configuration](config_environment.md) for `EVAL_JUDGE_MODEL` and credentials, [Dependencies](dependencies_libraries.md) for deepeval and rouge-score.

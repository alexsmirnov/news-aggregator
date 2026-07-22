import json
import logging

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import LiteLLMModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from metrics import match_groups, pairwise_prf, rouge_l

from news.digest.schemas import DigestRecord, NewsRecord

GROUPING_F1_MIN = 0.6
ROUGE_L_MIN = 0.3
JUDGE_MIN = 0.6
logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_grouping_pairwise_f1(
    grouping_run: tuple[str, str, list[NewsRecord]],
    expected_groups: list[dict[str, object]],
) -> None:
    # Arrange
    _, _, records = grouping_run

    # Act
    precision, recall, f1 = pairwise_prf(
        [{str(link) for link in record.links} for record in records],
        [_links(group) for group in expected_groups],
    )

    # Assert
    assert f1 >= GROUPING_F1_MIN, (
        f"grouping precision={precision:.3f}, recall={recall:.3f}, "
        f"f1={f1:.3f}"
    )


@pytest.mark.integration
def test_grouping_judge(
    grouping_run: tuple[str, str, list[NewsRecord]],
    expected_groups: list[dict[str, object]],
    judge: LiteLLMModel,
) -> None:
    # Arrange
    formatted, actual_json, _ = grouping_run
    metric = GEval(
        name="Grouping correctness",
        criteria=(
            "Determine whether the news items in the actual output are "
            "correctly grouped into trending events: items about the same "
            "real-world event must be in one group, and items about "
            "different events must not be merged. Judge only grouping "
            "correctness, not wording."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=JUDGE_MIN,
    )
    test_case = LLMTestCase(
        input=formatted,
        actual_output=actual_json,
        expected_output=json.dumps(expected_groups, indent=2),
    )

    # Act / Assert
    assert_test(test_case, [metric])


@pytest.mark.integration
def test_summary_rouge_l(
    grouping_run: tuple[str, str, list[NewsRecord]],
    refined_run: list[DigestRecord],
    expected_groups: list[dict[str, object]],
    expected_summaries: list[dict[str, object]],
) -> None:
    # Arrange
    _, _, records = grouping_run
    matched = _matched_summaries(
        records, refined_run, expected_groups, expected_summaries
    )

    # Act
    scores = [
        rouge_l(expected_summary, record.refined_summary or "")
        for record, expected_summary in matched
    ]

    # Assert
    assert scores, "no matched groups had expected summaries"
    mean = sum(scores) / len(scores)
    assert mean >= ROUGE_L_MIN, f"ROUGE-L mean={mean:.3f}, scores={scores}"


@pytest.mark.integration
async def test_summary_judge_mean(
    grouping_run: tuple[str, str, list[NewsRecord]],
    refined_run: list[DigestRecord],
    expected_groups: list[dict[str, object]],
    expected_summaries: list[dict[str, object]],
    judge: LiteLLMModel,
) -> None:
    # Arrange
    _, _, records = grouping_run
    matched = _matched_summaries(
        records, refined_run, expected_groups, expected_summaries
    )
    assert matched, "no matched groups had expected summaries"

    # Act
    scores = []
    for record, expected_summary in matched:
        metric = _summary_metric(judge)
        await metric.a_measure(
            LLMTestCase(
                input=record.title or "",
                actual_output=record.refined_summary or "",
                expected_output=expected_summary,
            )
        )
        if metric.score is None:
            raise AssertionError("summary judge returned no score")
        scores.append(metric.score)

    # Assert
    mean = sum(scores) / len(scores)
    assert mean >= JUDGE_MIN, f"judge mean={mean:.3f}, scores={scores}"


def _links(group: dict[str, object]) -> set[str]:
    links = group.get("links")
    if not isinstance(links, list) or not all(
        isinstance(link, str) for link in links
    ):
        raise ValueError("expected group links must be a list of strings")
    return set(links)


def _matched_summaries(
    records: list[NewsRecord],
    refined_records: list[DigestRecord],
    expected_groups: list[dict[str, object]],
    expected_summaries: list[dict[str, object]],
) -> list[tuple[DigestRecord, str]]:
    pairs = match_groups(
        [{str(link) for link in record.links} for record in records],
        [_links(group) for group in expected_groups],
    )
    assert pairs, "no predicted groups overlap expected groups"
    summaries = {
        item["title"]: item["summary"]
        for item in expected_summaries
        if isinstance(item.get("title"), str)
        and isinstance(item.get("summary"), str)
    }
    matched = []
    for predicted_index, expected_index in pairs:
        title = expected_groups[expected_index].get("title")
        summary = summaries.get(title)
        if not isinstance(summary, str):
            logger.warning("missing expected summary for %s", title)
            continue
        matched.append((refined_records[predicted_index], summary))
    return matched


def _summary_metric(judge: LiteLLMModel) -> GEval:
    return GEval(
        name="Summary faithfulness",
        criteria=(
            "Determine whether the actual summary faithfully covers the "
            "key facts of the expected summary without invented facts."
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        threshold=JUDGE_MIN,
    )

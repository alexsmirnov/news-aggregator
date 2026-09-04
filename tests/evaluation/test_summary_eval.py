import logging

import pytest
from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from metrics import links, match_groups, rouge_l

from conftest import JUDGE_MIN, ROUGE_L_MIN
from news.digest.schemas import DigestRecord, NewsRecord

logger = logging.getLogger(__name__)


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
    judge: GPTModel,
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


def _matched_summaries(
    records: list[NewsRecord],
    refined_records: list[DigestRecord],
    expected_groups: list[dict[str, object]],
    expected_summaries: list[dict[str, object]],
) -> list[tuple[DigestRecord, str]]:
    pairs = match_groups(
        [{str(link) for link in record.links} for record in records],
        [links(group) for group in expected_groups],
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


def _summary_metric(judge: GPTModel) -> GEval:
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

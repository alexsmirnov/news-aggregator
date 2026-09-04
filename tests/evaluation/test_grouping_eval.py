import json

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from metrics import links, pairwise_prf

from conftest import GROUPING_F1_MIN, JUDGE_MIN
from news.digest.schemas import NewsRecord


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
        [links(group) for group in expected_groups],
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
    judge: GPTModel,
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

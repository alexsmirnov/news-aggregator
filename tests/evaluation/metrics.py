from itertools import combinations

from rouge_score import rouge_scorer


def pairwise_prf(
    predicted: list[set[str]], expected: list[set[str]]
) -> tuple[float, float, float]:
    predicted_pairs = _cluster_pairs(predicted)
    expected_pairs = _cluster_pairs(expected)
    true_positives = len(predicted_pairs & expected_pairs)
    precision = (
        true_positives / len(predicted_pairs) if predicted_pairs else 0.0
    )
    recall = true_positives / len(expected_pairs) if expected_pairs else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def rouge_l(reference: str, candidate: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, candidate)["rougeL"].fmeasure


def match_groups(
    predicted: list[set[str]], expected: list[set[str]]
) -> list[tuple[int, int]]:
    candidates = sorted(
        (
            (
                _jaccard(predicted_links, expected_links),
                predicted_index,
                expected_index,
            )
            for predicted_index, predicted_links in enumerate(predicted)
            for expected_index, expected_links in enumerate(expected)
            if predicted_links & expected_links
        ),
        reverse=True,
    )
    matched_predicted: set[int] = set()
    matched_expected: set[int] = set()
    matches = []
    for _, predicted_index, expected_index in candidates:
        if (
            predicted_index in matched_predicted
            or expected_index in matched_expected
        ):
            continue
        matches.append((predicted_index, expected_index))
        matched_predicted.add(predicted_index)
        matched_expected.add(expected_index)
    return matches


def _cluster_pairs(groups: list[set[str]]) -> set[frozenset[str]]:
    return {
        frozenset(pair)
        for group in groups
        for pair in combinations(sorted(group), 2)
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)

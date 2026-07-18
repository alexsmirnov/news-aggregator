import pytest
from metrics import match_groups, pairwise_prf, rouge_l


def test_pairwise_prf_perfect_match() -> None:
    # Arrange
    groups = [{"a", "b"}]

    # Act
    result = pairwise_prf(groups, groups)

    # Assert
    assert result == (1.0, 1.0, 1.0)


def test_pairwise_prf_disjoint() -> None:
    # Arrange
    predicted = [{"a", "b"}]
    expected = [{"a"}, {"b"}]

    # Act
    result = pairwise_prf(predicted, expected)

    # Assert
    assert result == (0.0, 0.0, 0.0)


def test_pairwise_prf_partial_overlap() -> None:
    # Arrange
    predicted = [{"a", "b", "c"}]
    expected = [{"a", "b"}, {"c"}]

    # Act
    precision, recall, f1 = pairwise_prf(predicted, expected)

    # Assert
    assert precision == pytest.approx(1 / 3)
    assert recall == 1.0
    assert f1 == pytest.approx(0.5)


def test_pairwise_prf_empty_predicted() -> None:
    # Arrange
    expected = [{"a", "b"}]

    # Act
    result = pairwise_prf([], expected)

    # Assert
    assert result == (0.0, 0.0, 0.0)


def test_rouge_l_identical_texts() -> None:
    # Arrange
    text = "the cat sat on the mat"

    # Act
    score = rouge_l(text, text)

    # Assert
    assert score == pytest.approx(1.0)


def test_rouge_l_disjoint_texts() -> None:
    # Arrange
    reference = "alpha beta gamma"
    candidate = "delta epsilon zeta"

    # Act
    score = rouge_l(reference, candidate)

    # Assert
    assert score == 0.0


def test_match_groups_pairs_by_link_overlap() -> None:
    # Arrange
    predicted = [{"x", "y"}, {"w"}]
    expected = [{"x", "z"}, {"v"}]

    # Act
    pairs = match_groups(predicted, expected)

    # Assert
    assert pairs == [(0, 0)]

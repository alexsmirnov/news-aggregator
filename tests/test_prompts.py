import datetime

from news.digest.prompts import (
    grouping_system_prompt,
    grouping_user_prompt,
    refinement_system_prompt,
    refinement_user_prompt,
    trending_query,
)


def test_trending_query_text():
    # Act
    text = trending_query()

    # Assert
    assert text == (
        "What are the most trending US and world news for the last 24 hours"
    )


def test_grouping_system_prompt_embeds_trending_focus_and_uses_fixed_spelling():  # noqa: E501
    # Act
    text = grouping_system_prompt("TRENDING_EXAMPLES", "FOCUS_TEXT")

    # Assert
    assert "TRENDING_EXAMPLES" in text
    assert "Pay special attention to:\nFOCUS_TEXT" in text
    assert "happend" not in text
    assert "<news>" not in text
    assert "<title>" not in text
    assert "War in Ukraine" not in text


def test_grouping_user_prompt_wraps_content():
    # Act
    text = grouping_user_prompt("FORMATTED")

    # Assert
    assert text == "This is the news data\n<data>\nFORMATTED\n</data>"


def test_refinement_system_prompt_uses_fixed_spelling_and_date():
    # Act
    text = refinement_system_prompt(datetime.date(2026, 7, 17))

    # Assert
    assert "Today date is 2026-07-17" in text
    assert "# What happened" in text
    assert "# Contradictory opinions ( if any )" in text
    assert "hapenned" not in text
    assert "Contradictinal" not in text


def test_refinement_user_prompt_joins_links():
    # Act
    text = refinement_user_prompt("Title", "Summary", ["http://a", "http://b"])

    # Assert
    assert text == "# Title\n\nSummary\nsources: http://a http://b"

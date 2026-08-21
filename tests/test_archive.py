from datetime import date
from pathlib import Path

from news.digest.archive import (
    available_dates,
    build_page,
    digest_path,
    load_digest,
)
from news.digest.schemas import Digest, DigestRecord
from news.settings import Aggregation

AGGREGATIONS = [
    Aggregation(name="news", miniflux_category="news", focus="f"),
    Aggregation(name="economy", miniflux_category="Economy", focus="f"),
    Aggregation(name="technology", miniflux_category="Technology", focus="f"),
]


def write_digest_file(
    output_dir: Path, name: str, day: date, digest: Digest
) -> Path:
    path = digest_path(output_dir, name, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(digest.model_dump_json())
    return path


def make_digest(records: list[DigestRecord] | None = None) -> Digest:
    return Digest(
        generated_at="2026-08-21T12:00:00",
        records=records if records is not None else [],
    )


def test_digest_path_layout(tmp_path: Path) -> None:
    # Arrange / Act
    path = digest_path(tmp_path, "news", date(2026, 7, 5))

    # Assert
    assert path == tmp_path / "2026" / "07" / "news-05.json"


def test_available_dates_ordering(tmp_path: Path) -> None:
    # Arrange
    for day in (date(2026, 8, 19), date(2026, 8, 21), date(2026, 7, 30)):
        write_digest_file(tmp_path, "news", day, make_digest())

    # Act
    result = available_dates(tmp_path, "news")

    # Assert
    assert result == [date(2026, 7, 30), date(2026, 8, 19), date(2026, 8, 21)]


def test_available_dates_isolation(tmp_path: Path) -> None:
    # Arrange
    write_digest_file(tmp_path, "news", date(2026, 8, 21), make_digest())
    write_digest_file(tmp_path, "economy", date(2026, 8, 21), make_digest())

    # Act
    result = available_dates(tmp_path, "economy")

    # Assert
    assert result == [date(2026, 8, 21)]


def test_available_dates_prefix_collision(tmp_path: Path) -> None:
    # Arrange
    write_digest_file(tmp_path, "news", date(2026, 8, 21), make_digest())
    write_digest_file(tmp_path, "news-extra", date(2026, 8, 21), make_digest())

    # Act
    news_result = available_dates(tmp_path, "news")
    extra_result = available_dates(tmp_path, "news-extra")

    # Assert
    assert news_result == [date(2026, 8, 21)]
    assert extra_result == [date(2026, 8, 21)]


def test_available_dates_rejects_unpadded_day(tmp_path: Path) -> None:
    # Arrange
    stray = tmp_path / "2026" / "08" / "news-5.json"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text(make_digest().model_dump_json())

    # Act
    result = available_dates(tmp_path, "news")

    # Assert
    assert result == []


def test_available_dates_tolerates_malformed_entries(tmp_path: Path) -> None:
    # Arrange
    bad_day = tmp_path / "2026" / "08" / "news-XX.json"
    bad_day.parent.mkdir(parents=True, exist_ok=True)
    bad_day.write_text(make_digest().model_dump_json())

    bad_month = tmp_path / "2026" / "13" / "news-01.json"
    bad_month.parent.mkdir(parents=True, exist_ok=True)
    bad_month.write_text(make_digest().model_dump_json())

    stray_dir = tmp_path / "foo" / "bar" / "news-21.json"
    stray_dir.parent.mkdir(parents=True, exist_ok=True)
    stray_dir.write_text(make_digest().model_dump_json())

    write_digest_file(tmp_path, "news", date(2026, 8, 21), make_digest())

    # Act
    result = available_dates(tmp_path, "news")

    # Assert
    assert result == [date(2026, 8, 21)]


def test_available_dates_missing_directory(tmp_path: Path) -> None:
    # Arrange
    absent = tmp_path / "absent"

    # Act
    result = available_dates(absent, "news")

    # Assert
    assert result == []


def test_load_digest_round_trip(tmp_path: Path) -> None:
    # Arrange
    original = make_digest(
        [
            DigestRecord(
                title="Title",
                refined_summary="# Heading\n\n* bullet one\n* bullet two",
                links=["http://a.example", "http://b.example"],
            )
        ]
    )
    path = write_digest_file(tmp_path, "news", date(2026, 8, 21), original)

    # Act
    result = load_digest(path)

    # Assert
    assert result == original


def test_load_digest_truncated_file(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "broken.json"
    path.write_text('{"generated_')

    # Act
    result = load_digest(path)

    # Assert
    assert result is None


def test_load_digest_missing_file(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "missing.json"

    # Act
    result = load_digest(path)

    # Assert
    assert result is None


def test_build_page_default_aggregation(tmp_path: Path) -> None:
    # Arrange
    for agg in AGGREGATIONS:
        write_digest_file(
            tmp_path, agg.name, date(2026, 8, 21), make_digest()
        )

    # Act
    page = build_page(tmp_path, AGGREGATIONS, None, None)

    # Assert
    assert page is not None
    assert page.name == "news"


def test_build_page_latest_date(tmp_path: Path) -> None:
    # Arrange
    write_digest_file(
        tmp_path,
        "news",
        date(2026, 8, 19),
        make_digest([DigestRecord(title="Old", refined_summary="", links=[])]),
    )
    write_digest_file(
        tmp_path,
        "news",
        date(2026, 8, 21),
        make_digest([DigestRecord(title="New", refined_summary="", links=[])]),
    )

    # Act
    page = build_page(tmp_path, AGGREGATIONS, "news", None)

    # Assert
    assert page is not None
    assert page.day == date(2026, 8, 21)
    assert [r.title for r in page.records] == ["New"]


def test_build_page_unknown_name(tmp_path: Path) -> None:
    # Arrange / Act
    page = build_page(tmp_path, AGGREGATIONS, "sports", None)

    # Assert
    assert page is None


def test_build_page_missing_explicit_day(tmp_path: Path) -> None:
    # Arrange
    write_digest_file(tmp_path, "news", date(2026, 8, 21), make_digest())

    # Act
    page = build_page(tmp_path, AGGREGATIONS, "news", date(2020, 1, 1))

    # Assert
    assert page is None


def test_build_page_empty_archive(tmp_path: Path) -> None:
    # Arrange / Act
    page = build_page(tmp_path, AGGREGATIONS, "news", None)

    # Assert
    assert page is not None
    assert page.day is None
    assert page.generated_at is None
    assert page.records == []


def test_build_page_no_aggregations_configured(tmp_path: Path) -> None:
    # Arrange / Act
    page = build_page(tmp_path, [], None, None)

    # Assert
    assert page is not None
    assert page.day is None
    assert page.records == []


def test_build_page_neighbours(tmp_path: Path) -> None:
    # Arrange
    for day in (date(2026, 8, 19), date(2026, 8, 20), date(2026, 8, 21)):
        write_digest_file(tmp_path, "news", day, make_digest())

    # Act
    page = build_page(tmp_path, AGGREGATIONS, "news", date(2026, 8, 20))

    # Assert
    assert page is not None
    assert page.older is not None
    assert page.older.url == "/digest/news/2026-08-19"
    assert page.newer is not None
    assert page.newer.url == "/digest/news/2026-08-21"


def test_build_page_newest_date_has_no_newer(tmp_path: Path) -> None:
    # Arrange
    for day in (date(2026, 8, 19), date(2026, 8, 21)):
        write_digest_file(tmp_path, "news", day, make_digest())

    # Act
    page = build_page(tmp_path, AGGREGATIONS, "news", date(2026, 8, 21))

    # Assert
    assert page is not None
    assert page.newer is None
    assert page.older is not None


def test_build_page_oldest_date_has_no_older(tmp_path: Path) -> None:
    # Arrange
    for day in (date(2026, 8, 19), date(2026, 8, 21)):
        write_digest_file(tmp_path, "news", day, make_digest())

    # Act
    page = build_page(tmp_path, AGGREGATIONS, "news", date(2026, 8, 19))

    # Assert
    assert page is not None
    assert page.older is None
    assert page.newer is not None


def test_build_page_single_date_has_no_neighbours(tmp_path: Path) -> None:
    # Arrange
    write_digest_file(tmp_path, "news", date(2026, 8, 21), make_digest())

    # Act
    page = build_page(tmp_path, AGGREGATIONS, "news", date(2026, 8, 21))

    # Assert
    assert page is not None
    assert page.older is None
    assert page.newer is None


def test_build_page_aggregation_links(tmp_path: Path) -> None:
    # Arrange
    write_digest_file(tmp_path, "economy", date(2026, 8, 21), make_digest())

    # Act
    page = build_page(tmp_path, AGGREGATIONS, "economy", None)

    # Assert
    assert page is not None
    assert [link.url for link in page.aggregations] == [
        "/digest/news",
        "/digest/economy",
        "/digest/technology",
    ]
    assert [link.current for link in page.aggregations] == [
        False,
        True,
        False,
    ]


def test_build_page_null_title_and_summary(tmp_path: Path) -> None:
    # Arrange
    write_digest_file(
        tmp_path,
        "news",
        date(2026, 8, 21),
        make_digest(
            [DigestRecord(title=None, refined_summary=None, links=[])]
        ),
    )

    # Act
    page = build_page(tmp_path, AGGREGATIONS, "news", None)

    # Assert
    assert page is not None
    record = page.records[0]
    assert record.title == "Untitled"
    assert record.summary == ""
    assert record.title != "None"
    assert record.summary != "None"


def test_build_page_link_scheme_filter(tmp_path: Path) -> None:
    # Arrange
    write_digest_file(
        tmp_path,
        "news",
        date(2026, 8, 21),
        make_digest(
            [
                DigestRecord(
                    title="T",
                    refined_summary="",
                    links=[
                        "https://ok.example",
                        "javascript:alert(1)",
                        "http://ok2.example",
                    ],
                )
            ]
        ),
    )

    # Act
    page = build_page(tmp_path, AGGREGATIONS, "news", None)

    # Assert
    assert page is not None
    assert page.records[0].links == [
        "https://ok.example",
        "http://ok2.example",
    ]


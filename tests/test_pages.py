from datetime import date

from fastapi.testclient import TestClient
from tests.conftest import SeedDigest, body_text, title_text

from news.digest.schemas import Digest, DigestRecord


def make_digest(
    *titles: str, generated_at: str = "2026-08-21T00:00:00Z"
) -> Digest:
    return Digest(
        generated_at=generated_at,
        records=[
            DigestRecord(
                title=title,
                refined_summary=f"summary for {title}",
                links=[f"https://example.test/{title}"],
            )
            for title in titles
        ],
    )


def test_home_page_renders_default_aggregation(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 21), make_digest("News Item"))
    seed_digest("economy", date(2026, 8, 21), make_digest("Economy Item"))

    # Act
    response = client.get("/")

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "News Item" in response.text


def test_home_page_picks_newest_date(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 19), make_digest("Older Item"))
    seed_digest("news", date(2026, 8, 21), make_digest("Newer Item"))

    # Act
    response = client.get("/")

    # Assert
    assert "Newer Item" in response.text
    assert "Older Item" not in response.text


def test_aggregation_selection(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 21), make_digest("News Item"))
    seed_digest("economy", date(2026, 8, 21), make_digest("Economy Item"))

    # Act
    response = client.get("/digest/economy")

    # Assert
    assert response.status_code == 200
    assert "Economy Item" in response.text
    assert "News Item" not in response.text


def test_explicit_date(client: TestClient, seed_digest: SeedDigest) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 19), make_digest("Older Item"))
    seed_digest("news", date(2026, 8, 21), make_digest("Newer Item"))

    # Act
    response = client.get("/digest/news/2026-08-19")

    # Assert
    assert response.status_code == 200
    assert "Older Item" in response.text


def test_unknown_aggregation_returns_404(client: TestClient) -> None:
    # Act
    response = client.get("/digest/sports")

    # Assert
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert title_text(response.text) == "404 - News Aggregator"
    assert body_text(response.text) == "Page Not Found"


def test_missing_date_returns_404(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 21), make_digest("News Item"))

    # Act
    response = client.get("/digest/news/2020-01-01")

    # Assert
    assert response.status_code == 404
    assert title_text(response.text) == "404 - News Aggregator"


def test_malformed_date_returns_html_404_not_json(client: TestClient) -> None:
    # Act
    response = client.get("/digest/news/not-a-date")

    # Assert
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert title_text(response.text) == "404 - News Aggregator"


def test_numeric_date_segment_returns_404(client: TestClient) -> None:
    # Act
    response = client.get("/digest/news/1755739200")

    # Assert
    assert response.status_code == 404
    assert title_text(response.text) == "404 - News Aggregator"


def test_empty_state_returns_200_with_message(client: TestClient) -> None:
    # Act
    response = client.get("/")

    # Assert
    assert response.status_code == 200
    assert "No digests available yet" in response.text


def test_corrupt_digest_file_falls_back_to_empty_state(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    path = seed_digest("news", date(2026, 8, 21), make_digest("News Item"))
    path.write_text('{"generated_')

    # Act
    response = client.get("/")

    # Assert
    assert response.status_code == 200
    assert "No digests available yet" in response.text


def test_collapsible_rendering(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 21), make_digest("News Item"))

    # Act
    response = client.get("/")

    # Assert
    assert "<details" in response.text
    assert "<summary" in response.text
    assert "News Item" in response.text


def test_summary_is_escaped_not_interpreted(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    digest = Digest(
        generated_at="2026-08-21T00:00:00Z",
        records=[
            DigestRecord(
                title="Hostile",
                refined_summary="<script>alert(1)</script>",
                links=[],
            )
        ],
    )
    seed_digest("news", date(2026, 8, 21), digest)

    # Act
    response = client.get("/")

    # Assert
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;" in response.text


def test_links_open_in_new_tab(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 21), make_digest("News Item"))

    # Act
    response = client.get("/")

    # Assert
    assert "https://example.test/News Item" in response.text
    assert 'target="_blank"' in response.text
    assert 'rel="noopener noreferrer"' in response.text


def test_hostile_link_scheme_is_filtered(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    digest = Digest(
        generated_at="2026-08-21T00:00:00Z",
        records=[
            DigestRecord(
                title="Hostile",
                refined_summary="summary",
                links=["javascript:alert(1)"],
            )
        ],
    )
    seed_digest("news", date(2026, 8, 21), digest)

    # Act
    response = client.get("/")

    # Assert
    assert "javascript:" not in response.text


def test_null_title_falls_back_to_untitled(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    digest = Digest(
        generated_at="2026-08-21T00:00:00Z",
        records=[
            DigestRecord(title=None, refined_summary="summary", links=[])
        ],
    )
    seed_digest("news", date(2026, 8, 21), digest)

    # Act
    response = client.get("/")

    # Assert
    assert "<summary>Untitled</summary>" in response.text
    assert ">None<" not in response.text


def test_aggregation_nav_links(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("economy", date(2026, 8, 21), make_digest("Economy Item"))

    # Act
    response = client.get("/digest/economy")

    # Assert
    assert 'href="/digest/news"' in response.text
    assert 'href="/digest/technology"' in response.text


def test_older_newer_nav_present_between_dates(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 17), make_digest("Oldest"))
    seed_digest("news", date(2026, 8, 19), make_digest("Middle"))
    seed_digest("news", date(2026, 8, 21), make_digest("Newest"))

    # Act
    response = client.get("/digest/news/2026-08-19")

    # Assert
    assert 'href="/digest/news/2026-08-17"' in response.text
    assert 'href="/digest/news/2026-08-21"' in response.text


def test_nav_absent_past_newest_date(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 19), make_digest("Older"))
    seed_digest("news", date(2026, 8, 21), make_digest("Newest"))

    # Act
    response = client.get("/digest/news/2026-08-21")

    # Assert
    assert 'href="/digest/news/2026-08-22"' not in response.text
    assert "&rarr;" not in response.text


def test_nav_absent_on_single_date_archive(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 21), make_digest("Only Item"))

    # Act
    response = client.get("/")

    # Assert
    assert "/digest/news/" not in response.text


def test_head_supported_on_all_digest_routes(
    client: TestClient, seed_digest: SeedDigest
) -> None:
    # Arrange
    seed_digest("news", date(2026, 8, 21), make_digest("News Item"))

    # Act / Assert
    assert client.head("/").status_code == 200
    assert client.head("/digest/news").status_code == 200
    assert client.head("/digest/news/2026-08-21").status_code == 200


def test_trailing_slash_redirects_bare_prefix_404s(
    client: TestClient,
) -> None:
    # Act
    redirect = client.get("/digest/news/", follow_redirects=False)
    bare = client.get("/digest")

    # Assert
    assert redirect.status_code == 307
    assert bare.status_code == 404


def test_regression_guards_unaffected_routes(client: TestClient) -> None:
    # Act
    style = client.get("/style.css")
    favicon = client.get("/favicon.svg")
    missing = client.get("/does-not-exist")
    openapi = client.get("/openapi.json")

    # Assert
    assert style.status_code == 200
    assert style.headers["content-type"].startswith("text/css")
    assert favicon.status_code == 200
    assert favicon.headers["content-type"] == "image/svg+xml"
    assert missing.status_code == 404
    assert missing.headers["content-type"].startswith("text/html")
    assert "/" not in openapi.json()["paths"]

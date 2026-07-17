import re

from fastapi.testclient import TestClient


def body_text(html: str) -> str:
    body = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL)
    assert body is not None
    text = re.sub(r"<[^>]+>", "", body.group(1))
    return " ".join(text.split())


def title_text(html: str) -> str:
    title = re.search(r"<title>(.*?)</title>", html, re.DOTALL)
    assert title is not None
    return title.group(1).strip()


def test_home_page_returns_ok_html_without_redirect(
    client: TestClient,
) -> None:
    # Arrange
    path = "/"

    # Act
    response = client.get(path, follow_redirects=False)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_home_page_title_and_body_text(client: TestClient) -> None:
    # Arrange
    path = "/"

    # Act
    response = client.get(path)

    # Assert
    assert title_text(response.text) == "News Aggregator"
    assert body_text(response.text) == "Hello World"


def test_home_page_references_shared_stylesheet_and_favicon(
    client: TestClient,
) -> None:
    # Arrange
    path = "/"

    # Act
    response = client.get(path)

    # Assert
    assert 'href="/style.css"' in response.text
    assert 'href="/favicon.svg"' in response.text


def test_stylesheet_loads(client: TestClient) -> None:
    # Arrange
    path = "/style.css"

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_favicon_loads(client: TestClient) -> None:
    # Arrange
    path = "/favicon.svg"

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"


def test_undefined_path_returns_404_html_without_redirect(
    client: TestClient,
) -> None:
    # Arrange
    path = "/does-not-exist"

    # Act
    response = client.get(path, follow_redirects=False)

    # Assert
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")


def test_undefined_path_title_and_body_text(client: TestClient) -> None:
    # Arrange
    path = "/does-not-exist"

    # Act
    response = client.get(path)

    # Assert
    assert title_text(response.text) == "404 - News Aggregator"
    assert body_text(response.text) == "Page Not Found"


def test_undefined_path_references_shared_stylesheet_and_favicon(
    client: TestClient,
) -> None:
    # Arrange
    path = "/does-not-exist"

    # Act
    response = client.get(path)

    # Assert
    assert 'href="/style.css"' in response.text
    assert 'href="/favicon.svg"' in response.text


def test_missing_asset_path_returns_404_html(client: TestClient) -> None:
    # Arrange
    path = "/missing.css"

    # Act
    response = client.get(path, follow_redirects=False)

    # Assert
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert title_text(response.text) == "404 - News Aggregator"


def test_docs_endpoint_still_available(client: TestClient) -> None:
    # Arrange
    path = "/docs"

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200


def test_redoc_endpoint_still_available(client: TestClient) -> None:
    # Arrange
    path = "/redoc"

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200


def test_openapi_schema_still_available(client: TestClient) -> None:
    # Arrange
    path = "/openapi.json"

    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 200
    assert response.json() is not None

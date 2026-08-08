import pytest
from fastapi.testclient import TestClient

from news.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

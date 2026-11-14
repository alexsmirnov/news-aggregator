import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from group_editor import ExpectedGroup, GroupEditor
from group_editor_app import create_app
from group_editor_fixtures import (
    LINK_A,
    LINK_B,
    LINK_MISSING,
    make_entries,
    make_groups,
    saved,
)

MAX_GROUP_COUNT = 62


@pytest.fixture
def groups_path(tmp_path: Path) -> Path:
    return tmp_path / "expected_groups_test.yaml"


@pytest.fixture
def editor(groups_path: Path) -> GroupEditor:
    return GroupEditor(make_entries(), make_groups(), groups_path)


@pytest.fixture
def client(editor: GroupEditor) -> TestClient:
    return TestClient(create_app(editor))


def _keys_text(body: str, index: int) -> str | None:
    match = re.search(rf'id="keys-{index}"[^>]*>(.*?)<', body, re.S)
    return match.group(1).strip() if match else None


def test_page_lists_groups_and_entries(client: TestClient) -> None:
    response = client.get("/")

    body = response.text
    assert response.status_code == 200
    assert 'id="group-0"' in body
    assert "<kbd>0</kbd>" in body
    assert re.search(r"Group Zero.{0,80}\(1\)", body, re.S)
    assert 'id="group-1"' in body
    assert 'id="entry-0"' in body
    assert _keys_text(body, 0) == "0"
    assert "htmx" in body
    assert "line1" not in body


def _entry_block(body: str, index: int) -> str:
    start = body.index(f'id="entry-{index}"')
    end = body.find('id="entry-', start + 1)
    return body[start:] if end == -1 else body[start:end]


def _page_with_content(tmp_path: Path, content: str) -> str:
    entries = make_entries()
    entries[0] = entries[0].model_copy(update={"content": content})
    editor = GroupEditor(entries, make_groups(), tmp_path / "g.yaml")
    return TestClient(create_app(editor)).get("/").text


def test_page_shows_content_size_for_entries(client: TestClient) -> None:
    block = _entry_block(client.get("/").text, 0)

    assert re.search(r'class="ge-size"[^>]*>11<', block)
    assert "ge-empty" not in block


def test_page_marks_empty_content_entries(client: TestClient) -> None:
    block = _entry_block(client.get("/").text, 1)

    assert "ge-empty" in block
    assert "ge-size" not in block


def test_page_treats_whitespace_content_as_empty(tmp_path: Path) -> None:
    block = _entry_block(_page_with_content(tmp_path, "  \n "), 0)

    assert "ge-empty" in block
    assert "ge-size" not in block


def test_page_abbreviates_large_content_size(tmp_path: Path) -> None:
    block = _entry_block(_page_with_content(tmp_path, "x" * 1234), 0)

    assert re.search(r'class="ge-size"[^>]*>1\.2k<', block)


def test_page_has_entry_filter_input(client: TestClient) -> None:
    body = client.get("/").text

    assert re.search(r'<input[^>]*id="ge-filter"', body)
    assert 'id="ge-filter-count"' in body


def test_page_marks_missing_member(client: TestClient) -> None:
    body = client.get("/").text

    missing_row = rf'class="[^"]*\bmissing\b[^"]*"[^>]*{LINK_MISSING}'
    assert re.search(missing_row, body)


def test_page_escapes_titles(client: TestClient) -> None:
    body = client.get("/").text

    assert "Gamma &lt;b&gt;bold&lt;/b&gt; &amp; co" in body


def test_content_is_escaped(client: TestClient) -> None:
    response = client.get("/entries/2/content")

    assert response.status_code == 200
    assert "&lt;script&gt;x&lt;/script&gt;" in response.text
    assert "<script>x" not in response.text


def test_content_unknown_index_returns_404(client: TestClient) -> None:
    assert client.get("/entries/99/content").status_code == 404


def test_toggle_returns_oob_fragments_and_saves(
    client: TestClient, groups_path: Path
) -> None:
    response = client.post("/groups/1/toggle", params={"link": LINK_B})

    body = response.text
    assert response.status_code == 200
    assert re.search(r'id="group-1"[^>]*hx-swap-oob="true"', body)
    assert "Beta falls" in body
    assert _keys_text(body, 1) == "1"
    assert LINK_B in [e["link"] for e in saved(groups_path)[1]["entries"]]


def test_toggle_again_removes_entry(
    client: TestClient, groups_path: Path
) -> None:
    client.post("/groups/1/toggle", params={"link": LINK_B})

    response = client.post("/groups/1/toggle", params={"link": LINK_B})

    assert _keys_text(response.text, 1) == ""
    assert LINK_B not in [e["link"] for e in saved(groups_path)[1]["entries"]]


def test_move_returns_both_groups(client: TestClient) -> None:
    response = client.post(
        "/groups/0/move", params={"target": 1, "link": LINK_A}
    )

    body = response.text
    assert response.status_code == 200
    assert re.search(r'id="group-0"[^>]*hx-swap-oob="true"', body)
    assert re.search(r'id="group-1"[^>]*hx-swap-oob="true"', body)
    assert _keys_text(body, 0) == "1"


def test_move_same_group_returns_204(client: TestClient) -> None:
    response = client.post(
        "/groups/0/move", params={"target": 0, "link": LINK_A}
    )

    assert response.status_code == 204
    assert response.text == ""


def test_remove_from_group(client: TestClient, groups_path: Path) -> None:
    response = client.request(
        "DELETE", "/groups/1/entries", params={"link": LINK_MISSING}
    )

    assert response.status_code == 200
    assert re.search(r"Group One.{0,80}\(0\)", response.text, re.S)
    assert len(saved(groups_path)) == 2


def test_add_group_appends_oob(client: TestClient) -> None:
    response = client.post("/groups")

    body = response.text
    assert response.status_code == 200
    assert 'hx-swap-oob="beforeend:#groups"' in body
    assert 'id="group-2"' in body
    assert "<kbd>2</kbd>" in body
    assert "New group" in body


def test_add_group_at_limit_returns_204(groups_path: Path) -> None:
    groups = [ExpectedGroup(title=f"G{i}") for i in range(MAX_GROUP_COUNT)]
    full_client = TestClient(
        create_app(GroupEditor(make_entries(), groups, groups_path))
    )

    assert full_client.post("/groups").status_code == 204


def test_rename_returns_group(client: TestClient) -> None:
    response = client.post("/groups/0/title", params={"title": "Fresh"})

    assert response.status_code == 200
    assert re.search(r'id="group-0".*Fresh', response.text, re.S)


def test_rename_blank_returns_old_title(client: TestClient) -> None:
    response = client.post("/groups/0/title", params={"title": " "})

    assert response.status_code == 200
    assert "Group Zero" in response.text


def test_unknown_group_returns_404(client: TestClient) -> None:
    response = client.post("/groups/9/toggle", params={"link": LINK_A})

    assert response.status_code == 404


def test_unknown_link_returns_404(client: TestClient) -> None:
    response = client.post(
        "/groups/0/toggle", params={"link": "https://x/zzz"}
    )

    assert response.status_code == 404


def test_save_failure_returns_500_and_keeps_state(tmp_path: Path) -> None:
    directory = tmp_path / "dir.yaml"
    directory.mkdir()
    failing = TestClient(
        create_app(GroupEditor(make_entries(), make_groups(), directory))
    )

    response = failing.post("/groups/1/toggle", params={"link": LINK_B})

    assert response.status_code == 500
    assert response.json()["detail"].startswith("Save failed")
    assert re.search(r"Group One.{0,80}\(1\)", failing.get("/").text, re.S)


@pytest.mark.parametrize("asset", ["editor.js", "editor.css"])
def test_static_assets_are_served(client: TestClient, asset: str) -> None:
    assert client.get(f"/static/{asset}").status_code == 200

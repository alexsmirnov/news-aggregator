import json
from pathlib import Path

import pytest
import yaml
from group_editor import (
    ExpectedEntry,
    ExpectedGroup,
    GroupEditor,
    UnknownEntryError,
    UnknownGroupError,
    hotkey,
    load_entries,
    load_groups,
    save_groups,
)
from group_editor_fixtures import (
    LINK_A,
    LINK_B,
    LINK_MISSING,
    make_entries,
    make_groups,
    saved,
)
from pydantic import ValidationError

NEW_ENTRY_TITLE = "Beta falls"


@pytest.fixture
def groups_path(tmp_path: Path) -> Path:
    return tmp_path / "expected_groups_test.yaml"


@pytest.fixture
def editor(groups_path: Path) -> GroupEditor:
    return GroupEditor(make_entries(), make_groups(), groups_path)


def test_hotkey_maps_digits_then_lower_then_upper() -> None:
    assert [hotkey(i) for i in (0, 9, 10, 35, 36, 61, 62)] == [
        "0",
        "9",
        "a",
        "z",
        "A",
        "Z",
        "",
    ]


def test_load_groups_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_groups(tmp_path / "none.yaml") == []


def test_load_groups_reads_existing_format(groups_path: Path) -> None:
    raw_groups = [group.model_dump() for group in make_groups()]
    groups_path.write_text(yaml.safe_dump(raw_groups, sort_keys=False))

    assert load_groups(groups_path) == make_groups()


def test_load_groups_rejects_unknown_keys(groups_path: Path) -> None:
    groups_path.write_text("- title: T\n  entries: []\n  extra: 1\n")

    with pytest.raises(ValidationError):
        load_groups(groups_path)


def test_load_entries_rejects_duplicate_links(tmp_path: Path) -> None:
    path = tmp_path / "rss_entries_test.json"
    duplicated = [e.model_dump(mode="json") for e in make_entries()[:1] * 2]
    path.write_text(json.dumps(duplicated))

    with pytest.raises(ValueError, match=LINK_A):
        load_entries(path)


def test_save_groups_keeps_existing_format(tmp_path: Path) -> None:
    path = tmp_path / "out.yaml"
    title = "GM’s stock " + "x" * 120
    group = ExpectedGroup(
        title=title, entries=[ExpectedEntry(title="t", link=LINK_A)]
    )

    save_groups(path, [group])

    assert path.read_text() == (
        f"- title: {title}\n  entries:\n  - title: t\n    link: {LINK_A}\n"
    )
    assert [p.name for p in tmp_path.iterdir()] == ["out.yaml"]


def test_toggle_appends_entry_and_saves(
    editor: GroupEditor, groups_path: Path
) -> None:
    assert editor.toggle(1, LINK_B) is True

    assert saved(groups_path)[1]["entries"] == [
        {"title": "Old gone entry", "link": LINK_MISSING},
        {"title": NEW_ENTRY_TITLE, "link": LINK_B},
    ]


def test_toggle_existing_entry_removes_it(
    editor: GroupEditor, groups_path: Path
) -> None:
    editor.toggle(0, LINK_A)

    assert saved(groups_path)[0] == {"title": "Group Zero", "entries": []}


def test_entry_can_belong_to_several_groups(
    editor: GroupEditor, groups_path: Path
) -> None:
    editor.toggle(1, LINK_A)

    data = saved(groups_path)
    assert LINK_A in [e["link"] for e in data[0]["entries"]]
    assert LINK_A in [e["link"] for e in data[1]["entries"]]


def test_move_to_other_group(editor: GroupEditor, groups_path: Path) -> None:
    editor.move(0, 1, LINK_A)

    data = saved(groups_path)
    assert data[0]["entries"] == []
    assert data[1]["entries"][-1] == {"title": "Alpha rises", "link": LINK_A}


def test_move_into_group_already_holding_entry(
    editor: GroupEditor, groups_path: Path
) -> None:
    editor.toggle(1, LINK_A)

    editor.move(0, 1, LINK_A)

    data = saved(groups_path)
    assert data[0]["entries"] == []
    assert [e["link"] for e in data[1]["entries"]].count(LINK_A) == 1


def test_move_same_group_is_noop(
    editor: GroupEditor, groups_path: Path
) -> None:
    assert editor.move(0, 0, LINK_A) is False

    assert not groups_path.exists()


def test_move_entry_missing_from_dataset_keeps_its_title(
    editor: GroupEditor, groups_path: Path
) -> None:
    editor.move(1, 0, LINK_MISSING)

    assert saved(groups_path)[0]["entries"][-1]["title"] == "Old gone entry"


def test_remove_only_from_that_group(
    editor: GroupEditor, groups_path: Path
) -> None:
    editor.remove(1, LINK_MISSING)

    data = saved(groups_path)
    assert data[1]["entries"] == []
    assert len(data) == 2
    assert data[0]["entries"] == [{"title": "Alpha rises", "link": LINK_A}]


def test_add_group_appends_new_group(
    editor: GroupEditor, groups_path: Path
) -> None:
    assert editor.add_group() == 2

    assert saved(groups_path)[-1] == {"title": "New group", "entries": []}


def test_add_group_stops_at_hotkey_limit(
    editor: GroupEditor, groups_path: Path
) -> None:
    for _ in range(60):
        editor.add_group()

    assert editor.add_group() is None
    assert len(saved(groups_path)) == 62


def test_rename_trims_title(editor: GroupEditor, groups_path: Path) -> None:
    assert editor.rename(0, "  Fresh  ") is True

    assert saved(groups_path)[0]["title"] == "Fresh"


def test_rename_blank_keeps_old_title(
    editor: GroupEditor, groups_path: Path
) -> None:
    assert editor.rename(0, "  ") is False

    assert not groups_path.exists()


@pytest.mark.parametrize(
    "call",
    [
        lambda e: e.toggle(9, LINK_A),
        lambda e: e.move(0, 9, LINK_A),
        lambda e: e.rename(9, "x"),
    ],
)
def test_unknown_group_raises(editor: GroupEditor, call) -> None:
    with pytest.raises(UnknownGroupError):
        call(editor)


@pytest.mark.parametrize(
    "call",
    [
        lambda e: e.toggle(0, "https://x/zzz"),
        lambda e: e.remove(0, LINK_B),
    ],
)
def test_unknown_link_raises(editor: GroupEditor, call) -> None:
    with pytest.raises(UnknownEntryError):
        call(editor)


def test_missing_file_created_on_first_change(tmp_path: Path) -> None:
    path = tmp_path / "new.yaml"

    GroupEditor(make_entries(), [], path).add_group()

    assert saved(path) == [{"title": "New group", "entries": []}]


def test_failed_save_keeps_state(tmp_path: Path) -> None:
    directory = tmp_path / "dir.yaml"
    directory.mkdir()
    editor = GroupEditor(make_entries(), make_groups(), directory)

    with pytest.raises(OSError):
        editor.toggle(1, LINK_B)

    assert len(editor.groups[1].entries) == 1


def test_assigned_keys_lists_all_groups_of_entry(editor: GroupEditor) -> None:
    editor.toggle(1, LINK_A)

    assert editor.assigned_keys(LINK_A) == "01"

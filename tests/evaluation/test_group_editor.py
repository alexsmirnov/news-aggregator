import json
from pathlib import Path

import pytest
import yaml
from group_editor import (
    MAX_GROUPS,
    ExpectedEntry,
    ExpectedGroup,
    GroupEditor,
    UnknownEntryError,
    UnknownGroupError,
    assign_hotkeys,
    load_entries,
    load_groups,
    save_groups,
    sort_by_trend,
)
from group_editor_fixtures import (
    CLUSTER_YAML,
    LINK_A,
    LINK_B,
    LINK_C,
    LINK_MISSING,
    make_cluster_groups,
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


@pytest.fixture
def cluster_editor(groups_path: Path) -> GroupEditor:
    return GroupEditor(make_entries(), make_cluster_groups(), groups_path)


def test_assign_hotkeys_up_to_62_are_single_characters() -> None:
    keys = assign_hotkeys(62)

    assert keys[:3] == ["0", "1", "2"]
    assert [keys[i] for i in (9, 10, 35, 36, 61)] == ["9", "a", "z", "A", "Z"]
    assert assign_hotkeys(3) == ["0", "1", "2"]


def test_assign_hotkeys_over_62_spills_into_two_characters() -> None:
    keys = assign_hotkeys(63)

    assert keys[60:] == ["Y", "Z0", "Z1"]


def test_assign_hotkeys_large_layout() -> None:
    keys = assign_hotkeys(576)

    assert keys[52] == "Q"
    assert keys[53] == "R0"
    assert keys[575] == "Zq"
    assert len(set(keys)) == 576


@pytest.mark.parametrize("count", [63, 576, MAX_GROUPS])
def test_assign_hotkeys_is_prefix_free(count: int) -> None:
    keys = assign_hotkeys(count)

    singles = {key for key in keys if len(key) == 1}
    prefixes = {key[0] for key in keys if len(key) == 2}
    assert not singles & prefixes
    assert len(set(keys)) == count


def test_max_groups_uses_every_two_character_key() -> None:
    assert MAX_GROUPS == 3783
    assert len(assign_hotkeys(MAX_GROUPS)) == MAX_GROUPS


def test_load_groups_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_groups(tmp_path / "none.yaml") == []


def test_load_groups_reads_existing_format(groups_path: Path) -> None:
    raw_groups = [group.model_dump() for group in make_groups()]
    groups_path.write_text(yaml.safe_dump(raw_groups, sort_keys=False))

    assert load_groups(groups_path) == make_groups()


def test_load_groups_accepts_cluster_file(groups_path: Path) -> None:
    groups_path.write_text(CLUSTER_YAML)

    groups = load_groups(groups_path)

    assert len(groups) == 3
    assert groups[1].title is None
    assert groups[1].model_extra == {
        "size": 2,
        "novelty": None,
        "indices": [0, 1],
        "trend_score": 0.9,
    }


def test_load_groups_rejects_unknown_entry_keys(groups_path: Path) -> None:
    groups_path.write_text(
        "- title: T\n  entries:\n  - title: t\n    link: l\n    extra: 1\n"
    )

    with pytest.raises(ValidationError):
        load_groups(groups_path)


def test_sort_by_trend_descending_then_unscored_stable() -> None:
    ordered = sort_by_trend(make_cluster_groups())

    assert [(g.model_extra or {}).get("trend_score") for g in ordered] == [
        0.9,
        0.5,
        None,
    ]


def test_sort_by_trend_keeps_order_without_scores() -> None:
    assert sort_by_trend(make_groups()) == make_groups()


def test_sort_by_trend_ignores_non_numeric_scores() -> None:
    groups = [
        ExpectedGroup.model_validate({"trend_score": "high", "title": "a"}),
        ExpectedGroup.model_validate({"trend_score": True, "title": "b"}),
        ExpectedGroup.model_validate({"trend_score": 1, "title": "c"}),
    ]

    assert [g.title for g in sort_by_trend(groups)] == ["c", "a", "b"]


def test_save_groups_keeps_metadata_and_order(tmp_path: Path) -> None:
    path = tmp_path / "out.yaml"

    save_groups(path, make_cluster_groups())

    result = saved(path)
    assert list(result[1]) == [
        "size",
        "novelty",
        "indices",
        "trend_score",
        "entries",
    ]
    assert result[1]["novelty"] is None
    assert result[1]["indices"] == [0, 1]
    assert list(result[2]) == ["title", "entries"]
    assert result[2]["title"] == ""


def test_editor_saves_groups_sorted_with_metadata(
    cluster_editor: GroupEditor, groups_path: Path
) -> None:
    assert cluster_editor.toggle(2, LINK_A) is True

    result = saved(groups_path)
    assert [g.get("trend_score") for g in result] == [0.9, 0.5, None]
    assert result[0]["indices"] == [0, 1]


def test_rename_untitled_group_puts_title_first(
    cluster_editor: GroupEditor, groups_path: Path
) -> None:
    assert cluster_editor.rename(0, "Fresh") is True

    first = saved(groups_path)[0]
    assert list(first)[0] == "title"
    assert first["title"] == "Fresh"
    assert first["indices"] == [0, 1]


def test_rename_untitled_blank_writes_nothing(groups_path: Path) -> None:
    groups_path.write_text(CLUSTER_YAML)
    cluster_editor = GroupEditor(
        make_entries(), load_groups(groups_path), groups_path
    )

    assert cluster_editor.rename(0, " ") is False

    assert groups_path.read_text() == CLUSTER_YAML


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


def test_add_group_stops_at_max_groups(groups_path: Path) -> None:
    groups = [ExpectedGroup(title=f"G{i}") for i in range(MAX_GROUPS - 1)]
    full = GroupEditor(make_entries(), groups, groups_path)

    assert full.add_group() == MAX_GROUPS - 1
    assert full.add_group() is None
    assert len(saved(groups_path)) == MAX_GROUPS


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

    assert editor.assigned_keys(LINK_A) == "0 1"


def test_assigned_keys_use_two_character_keys_past_62_groups(
    groups_path: Path,
) -> None:
    groups = [ExpectedGroup(title=f"G{i}") for i in range(63)]
    big = GroupEditor(make_entries(), groups, groups_path)
    big.toggle(0, LINK_C)
    big.toggle(62, LINK_C)

    assert big.assigned_keys(LINK_C) == "0 Z1"
    assert big.key_of(62) == "Z1"

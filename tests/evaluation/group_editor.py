import json
import math
import os
import sys
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from news.digest.schemas import RssEntry

HOTKEYS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
# One single-character key must stay free, every other character can be a
# prefix of HOTKEYS-many two-character keys.
MAX_GROUPS = 1 + (len(HOTKEYS) - 1) * len(HOTKEYS)
NEW_GROUP_TITLE = "New group"
TREND_SCORE_KEY = "trend_score"


class ExpectedEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    link: str


class ExpectedGroup(BaseModel):
    """Expected group; extra keys (cluster metrics) are kept but not shown."""

    model_config = ConfigDict(extra="allow")

    title: str | None = None
    entries: list[ExpectedEntry] = []


class UnknownGroupError(LookupError):
    """Group index is outside the current group list."""


class UnknownEntryError(LookupError):
    """Entry link is not available for the requested operation."""


_GROUPS_ADAPTER = TypeAdapter(list[ExpectedGroup])
_YAML_LINE_WIDTH = 1_000_000


def assign_hotkeys(count: int) -> list[str]:
    """Return a prefix-free key for each of `count` groups.

    Up to len(HOTKEYS) groups get one character each. Beyond that, the
    last characters of HOTKEYS become prefixes of two-character keys and
    only as many of them as needed are taken from the single-key pool.
    """
    if not 0 <= count <= MAX_GROUPS:
        raise ValueError(f"group count out of range: {count}")
    single_count = len(HOTKEYS)
    if count <= single_count:
        return list(HOTKEYS[:count])
    prefix_count = math.ceil((count - single_count) / (single_count - 1))
    singles = HOTKEYS[: single_count - prefix_count]
    two_character = (
        prefix + second
        for prefix in HOTKEYS[single_count - prefix_count :]
        for second in HOTKEYS
    )
    keys = list(singles)
    keys.extend(key for _, key in zip(range(count - len(keys)), two_character))
    return keys


def is_untitled(group: ExpectedGroup) -> bool:
    return not (group.title or "").strip()


def _trend_score(group: ExpectedGroup) -> float | None:
    score = (group.model_extra or {}).get(TREND_SCORE_KEY)
    numeric = isinstance(score, int | float) and not isinstance(score, bool)
    return score if numeric else None


def sort_by_trend(groups: list[ExpectedGroup]) -> list[ExpectedGroup]:
    """Highest trend score first; groups without a numeric score keep order."""

    def sort_key(group: ExpectedGroup) -> tuple[int, float]:
        score = _trend_score(group)
        return (1, 0.0) if score is None else (0, -score)

    return sorted(groups, key=sort_key)


def load_entries(path: Path) -> list[RssEntry]:
    raw_entries = json.loads(path.read_text())
    entries = [RssEntry.model_validate(raw) for raw in raw_entries]
    link_counts = Counter(entry.link for entry in entries)
    duplicates = [link for link, count in link_counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate entry links: {', '.join(duplicates)}")
    return entries


def load_groups(path: Path) -> list[ExpectedGroup]:
    if not path.exists():
        return []
    raw_groups = yaml.safe_load(path.read_text()) or []
    return _GROUPS_ADAPTER.validate_python(raw_groups)


def _group_to_yaml(group: ExpectedGroup) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if group.title is not None:
        data["title"] = group.title
    data.update(group.model_extra or {})
    data["entries"] = [entry.model_dump() for entry in group.entries]
    return data


def save_groups(path: Path, groups: list[ExpectedGroup]) -> None:
    text = yaml.safe_dump(
        [_group_to_yaml(group) for group in groups],
        sort_keys=False,
        allow_unicode=True,
        width=_YAML_LINE_WIDTH,
    )
    temp_path = path.with_name(path.name + ".tmp")
    try:
        temp_path.write_text(text)
        os.replace(temp_path, path)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise


def _find(group: ExpectedGroup, link: str) -> int | None:
    links = [entry.link for entry in group.entries]
    return links.index(link) if link in links else None


class GroupEditor:
    entry_index: dict[str, int]

    def __init__(
        self,
        entries: list[RssEntry],
        groups: list[ExpectedGroup],
        groups_path: Path,
    ) -> None:
        self.entries = entries
        self.groups = sort_by_trend(groups)
        self.groups_path = groups_path
        self.entry_index = {}
        for position, entry in enumerate(entries):
            self.entry_index.setdefault(entry.link, position)
        self._lock = threading.Lock()
        self._keys: list[str] = []

    def key_of(self, index: int) -> str:
        if len(self._keys) != len(self.groups):
            self._keys = assign_hotkeys(len(self.groups))
        return self._keys[index]

    def toggle(self, group: int, link: str) -> bool:
        def mutate(groups: list[ExpectedGroup]) -> bool:
            if link not in self.entry_index:
                raise UnknownEntryError(link)
            members = groups[group].entries
            position = _find(groups[group], link)
            if position is None:
                title = self.entries[self.entry_index[link]].title
                members.append(ExpectedEntry(title=title, link=link))
            else:
                del members[position]
            return True

        return self._change([group], mutate)

    def move(self, source: int, target: int, link: str) -> bool:
        def mutate(groups: list[ExpectedGroup]) -> bool:
            if source == target:
                return False
            position = _find(groups[source], link)
            if position is None:
                raise UnknownEntryError(link)
            entry = groups[source].entries.pop(position)
            if _find(groups[target], link) is None:
                groups[target].entries.append(entry)
            return True

        return self._change([source, target], mutate)

    def remove(self, group: int, link: str) -> bool:
        def mutate(groups: list[ExpectedGroup]) -> bool:
            position = _find(groups[group], link)
            if position is None:
                raise UnknownEntryError(link)
            del groups[group].entries[position]
            return True

        return self._change([group], mutate)

    def rename(self, group: int, title: str) -> bool:
        new_title = title.strip()

        def mutate(groups: list[ExpectedGroup]) -> bool:
            if not new_title or new_title == groups[group].title:
                return False
            groups[group].title = new_title
            return True

        return self._change([group], mutate)

    def add_group(self) -> int | None:
        with self._lock:
            if len(self.groups) >= MAX_GROUPS:
                return None
            updated = self._copy_groups()
            updated.append(ExpectedGroup(title=NEW_GROUP_TITLE))
            save_groups(self.groups_path, updated)
            self.groups = updated
            return len(updated) - 1

    def assigned_keys(self, link: str) -> str:
        return " ".join(
            self.key_of(i)
            for i, group in enumerate(self.groups)
            if _find(group, link) is not None
        )

    def _copy_groups(self) -> list[ExpectedGroup]:
        return [group.model_copy(deep=True) for group in self.groups]

    def _change(
        self,
        group_indexes: list[int],
        mutate: Callable[[list[ExpectedGroup]], bool],
    ) -> bool:
        with self._lock:
            for index in group_indexes:
                if not 0 <= index < len(self.groups):
                    raise UnknownGroupError(index)
            updated = self._copy_groups()
            if not mutate(updated):
                return False
            save_groups(self.groups_path, updated)
            self.groups = updated
            return True

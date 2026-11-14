import json
import os
import sys
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from news.digest.schemas import RssEntry

HOTKEYS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_GROUPS = len(HOTKEYS)
NEW_GROUP_TITLE = "New group"


class ExpectedEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    link: str


class ExpectedGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    entries: list[ExpectedEntry] = []


class UnknownGroupError(LookupError):
    """Group index is outside the current group list."""


class UnknownEntryError(LookupError):
    """Entry link is not available for the requested operation."""


_GROUPS_ADAPTER = TypeAdapter(list[ExpectedGroup])
_YAML_LINE_WIDTH = 1_000_000


def hotkey(index: int) -> str:
    return HOTKEYS[index] if 0 <= index < MAX_GROUPS else ""


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


def save_groups(path: Path, groups: list[ExpectedGroup]) -> None:
    text = yaml.safe_dump(
        [group.model_dump() for group in groups],
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
        self.groups = groups
        self.groups_path = groups_path
        self.entry_index = {}
        for position, entry in enumerate(entries):
            self.entry_index.setdefault(entry.link, position)
        self._lock = threading.Lock()

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
        return "".join(
            hotkey(i)
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

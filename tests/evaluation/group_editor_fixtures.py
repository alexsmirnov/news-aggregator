from pathlib import Path
from typing import Any

import yaml
from group_editor import ExpectedEntry, ExpectedGroup

from news.digest.schemas import RssEntry

LINK_A = "https://x/a"
LINK_B = "https://x/b"
LINK_C = "https://x/c"
LINK_MISSING = "https://x/missing"


def _entry(id_: int, title: str, link: str, content: str) -> RssEntry:
    return RssEntry(
        id=id_,
        title=title,
        link=link,
        content=content,
        published_at="2026-07-21T00:00:00Z",
        source="S",
    )


def make_entries() -> list[RssEntry]:
    return [
        _entry(1, "Alpha rises", LINK_A, "line1\nline2"),
        _entry(2, "Beta falls", LINK_B, ""),
        _entry(3, "Gamma <b>bold</b> & co", LINK_C, "<script>x</script>"),
    ]


def make_groups() -> list[ExpectedGroup]:
    return [
        ExpectedGroup(
            title="Group Zero",
            entries=[ExpectedEntry(title="Alpha rises", link=LINK_A)],
        ),
        ExpectedGroup(
            title="Group One",
            entries=[ExpectedEntry(title="Old gone entry", link=LINK_MISSING)],
        ),
    ]


def saved(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text())

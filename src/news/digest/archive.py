import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from news.digest.schemas import (
    Digest,
    DigestPage,
    DigestRecord,
    NavLink,
    RecordView,
)
from news.settings import Aggregation

logger = logging.getLogger(__name__)

_LINK_SCHEMES = ("http", "https")

# ponytail: read-only archive module combines repo+query for a ~100-line,
# DB-less filesystem reader; splitting further would add indirection with
# no benefit.


def digest_path(output_dir: Path, name: str, day: date) -> Path:
    return (
        output_dir
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"{name}-{day:%d}.json"
    )


def available_dates(output_dir: Path, name: str) -> list[date]:
    if not output_dir.is_dir():
        return []
    prefix = f"{name}-"
    dates: list[date] = []
    for path in output_dir.glob("*/*/*.json"):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        try:
            candidate = date(
                int(path.parent.parent.name),
                int(path.parent.name),
                int(stem[len(prefix) :]),
            )
        except ValueError:
            continue
        if digest_path(output_dir, name, candidate) == path:
            dates.append(candidate)
    return sorted(dates)


def load_digest(path: Path) -> Digest | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    try:
        return Digest.model_validate_json(text)
    except ValidationError:
        logger.warning("unreadable digest file path=%s", path, exc_info=True)
        return None


def _to_record_view(record: DigestRecord) -> RecordView:
    title = (record.title or "").strip() or "Untitled"
    summary = record.refined_summary or ""
    links = [
        link
        for link in record.links
        if urlsplit(link).scheme in _LINK_SCHEMES
    ]
    return RecordView(title=title, summary=summary, links=links)


def _empty_page(name: str, aggregations: list[NavLink]) -> DigestPage:
    return DigestPage(
        name=name,
        day=None,
        generated_at=None,
        records=[],
        aggregations=aggregations,
        older=None,
        newer=None,
    )


def build_page(
    output_dir: Path,
    aggregations: Sequence[Aggregation],
    name: str | None,
    day: date | None,
) -> DigestPage | None:
    agg_names = [agg.name for agg in aggregations]
    if name is None:
        if not agg_names:
            return _empty_page("", [])
        name = agg_names[0]
    elif name not in agg_names:
        return None

    nav_aggregations = [
        NavLink(label=n, url=f"/digest/{n}", current=n == name)
        for n in agg_names
    ]

    explicit_day = day is not None
    dates = available_dates(output_dir, name)
    if day is None:
        if not dates:
            return _empty_page(name, nav_aggregations)
        day = dates[-1]

    digest = load_digest(digest_path(output_dir, name, day))
    if digest is None:
        if explicit_day:
            return None
        return _empty_page(name, nav_aggregations)

    older_date = max((d for d in dates if d < day), default=None)
    newer_date = min((d for d in dates if d > day), default=None)
    older = (
        NavLink(
            label=older_date.isoformat(),
            url=f"/digest/{name}/{older_date.isoformat()}",
        )
        if older_date
        else None
    )
    newer = (
        NavLink(
            label=newer_date.isoformat(),
            url=f"/digest/{name}/{newer_date.isoformat()}",
        )
        if newer_date
        else None
    )

    return DigestPage(
        name=name,
        day=day,
        generated_at=digest.generated_at,
        records=[_to_record_view(r) for r in digest.records],
        aggregations=nav_aggregations,
        older=older,
        newer=newer,
    )

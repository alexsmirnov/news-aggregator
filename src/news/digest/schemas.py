from datetime import date

from pydantic import BaseModel, Field


class RssEntry(BaseModel):
    id: int
    title: str
    link: str
    content: str
    published_at: str
    source: str


class NewsRecord(BaseModel):
    title: str | None = Field(description="combined news headline")
    links: list[str] = Field(description="source pages for the news item")


class NewsResponse(BaseModel):
    records: list[NewsRecord] = Field(description="most popular breaking news")


class ClusterSummary(BaseModel):
    title: str = Field(description="combined news headline, in English")
    summary: str = Field(description="one to two sentence synthesis")


class MergedGroup(BaseModel):
    title: str = Field(description="combined news headline, in English")
    summary: str = Field(description="one to two sentence synthesis")
    member_indexes: list[int] = Field(
        description="1-based indexes of the input entries this group merges"
    )


class MergeResponse(BaseModel):
    groups: list[MergedGroup] = Field(
        description="entries merged by shared real-world event"
    )


class DigestRecord(BaseModel):
    title: str | None
    refined_summary: str | None
    links: list[str]


class Digest(BaseModel):
    generated_at: str
    records: list[DigestRecord]


class NavLink(BaseModel):
    label: str
    url: str
    current: bool = False


class RecordView(BaseModel):
    title: str
    summary: str
    links: list[str]


class DigestPage(BaseModel):
    name: str
    day: date | None
    generated_at: str | None
    records: list[RecordView]
    aggregations: list[NavLink]
    older: NavLink | None
    newer: NavLink | None

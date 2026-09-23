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
    entities: list[str] = Field(
        description="named people, organizations and locations, in English"
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

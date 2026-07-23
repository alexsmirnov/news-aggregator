from pydantic import BaseModel


class RssEntry(BaseModel):
    id: int
    title: str
    link: str
    content: str
    published_at: str
    source: str


class NewsRecord(BaseModel):
    title: str | None
    links: list[str]


class NewsResponse(BaseModel):
    records: list[NewsRecord]


class DigestRecord(BaseModel):
    title: str | None
    refined_summary: str | None
    links: list[str]


class Digest(BaseModel):
    generated_at: str
    records: list[DigestRecord]

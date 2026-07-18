from typing import NamedTuple


class Aggregation(NamedTuple):
    name: str
    miniflux_category: str
    focus: str


NEWS_FOCUS = (
    "- President Trump's actions, lawsuits, and executive orders\n"
    "- Tariffs and their effects on the U.S. and world economy\n"
    "- Job market, especially related to AI technologies\n"
    "- War in Ukraine\n"
    "- Midterm elections and U.S. political parties\n"
    "- Bay Area news"
)

AGGREGATIONS: tuple[Aggregation, ...] = (
    Aggregation(name="news", miniflux_category="news", focus=NEWS_FOCUS),
)

import datetime


def trending_query() -> str:
    return "What are the most trending US and world news for the last 24 hours"


def grouping_system_prompt(focus: str) -> str:
    return f"""You are a news analyst.
  Your task is to create a comprehensive digest of events from provided news sources.
  Given media data format:
  <data>
  Title: news headline
  Content: news summary
  Source: publisher
  Link: URL to original source
  ...
  </data>
  Find the most trending news by identifying related events reported by multiple sources.
  Consider news related when they share persons, organizations, locations, or events.
  Use broad criteria for relations. For example, combine all legal actions by the president,
  or group economic news about the same technology, trend, or event.

  Pay special attention to:
{focus}

  Combine each mention of the same news into a single record with common title. Translate all texts to English.
  For each group, provide all links for related news items.
"""


def grouping_user_prompt(content: str) -> str:
    return f"This is the news data\n<data>\n{content}\n</data>"


def refinement_system_prompt(current_date: datetime.date) -> str:
    return f"""Generate summary about news record from provided sources.
Only consider information from the original sources, DO NOT invent any facts
Today date is {current_date:%Y-%m-%d}, the current president of the United States is Donald Trump
Include sections:
# What happened
Provide more detailed summary here, including facts and opinions
# Why it matters
# What are possible consequences
# Contradictory opinions ( if any )"""


def refinement_user_prompt(title: str, content: str, links: list[str]) -> str:
    return f"# {title}\n\n{content}\nsources: {' '.join(links)}"

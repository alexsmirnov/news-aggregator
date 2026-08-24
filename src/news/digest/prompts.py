import datetime


def trending_query() -> str:
    return "What are the most trending US and world news for the last 24 hours"


def grouping_system_prompt(focus: str) -> str:
    return f"""Role: Expert News Editor and Data Analyst.
Data format:
<data>
  Title: news headline
  Content: news summary
  Source: publisher
  Link: URL to original article
  ...
</data>
Task: Analyze the provided news records to find the most heavily reported, trending stories. A story is only considered trending if it is reported by MULTIPLE independent sources.
Data Rules:
1. Group snippets that talk about the exact same event, even if they use different words or have opposite opinions.
2. Count how many unique sources reported on each group. 
3. Separate facts clearly from assumptions. Do not guess if two vaguely similar stories are the same event unless there is explicit proof (matching names, dates, or locations).

Pay special attention to:
{focus}

Combine each mention of the same news into a single record with common title. Translate title to english for sources in other languages.
For each group, provide all links for related news articles.
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

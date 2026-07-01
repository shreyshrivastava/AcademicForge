import logging

from backend.cache import cache_get, cache_set, make_cache_key
from backend.llm import generate_text
from backend.llm import model_name


SUMMARY_CACHE = {}
logger = logging.getLogger(__name__)


def _paper_cache_key(paper):
    return paper.get("paper_id") or paper.get("url") or paper.get("link") or paper["title"]


def summarize_paper(paper):
    """Generate a concise, useful research summary for one paper."""
    paper_id = _paper_cache_key(paper)
    cache_key = make_cache_key(
        "summary-v1",
        model_name("summary"),
        paper_id,
        paper.get("title"),
        paper.get("abstract"),
    )
    if cache_key in SUMMARY_CACHE:
        logger.info("Summary cache hit cache=memory paper_id=%r", paper_id)
        return SUMMARY_CACHE[cache_key]
    cached_summary = cache_get("summaries", cache_key)
    if cached_summary:
        logger.info("Summary cache hit cache=disk paper_id=%r", paper_id)
        SUMMARY_CACHE[cache_key] = cached_summary
        return cached_summary

    logger.info("Summary generation started paper_id=%r", paper_id)
    title = paper["title"]
    authors = ", ".join(paper.get("authors", []))
    abstract = paper["abstract"]

    system_prompt = (
        "You are AcademicForge, a careful academic research assistant. "
        "Summarize papers for builders who need to understand the idea, "
        "method, implementation implications, and limitations. Be specific, "
        "avoid hype, and do not invent details not supported by the abstract."
    )
    user_prompt = f"""
Paper title: {title}
Authors: {authors}
Abstract:
{abstract}

Write a structured summary with these sections:
- Core idea
- Method
- Why it matters
- Implementation notes
- Limitations or unknowns

Keep it concise but useful.
""".strip()

    summary = generate_text(system_prompt, user_prompt, token_budget=650, task="summary")
    SUMMARY_CACHE[cache_key] = summary
    cache_set("summaries", cache_key, summary)
    logger.info("Summary generation completed paper_id=%r", paper_id)
    return summary

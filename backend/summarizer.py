import logging

from backend.cache import cache_get, cache_set, make_cache_key
from backend.llm import generate_text
from backend.llm import model_name


SUMMARY_CACHE = {}
logger = logging.getLogger(__name__)


def _paper_cache_key(paper):
    return paper.get("paper_id") or paper.get("url") or paper.get("link") or paper["title"]


def summarize_paper(paper, model=None):
    """Generate a concise, useful research summary for one paper."""
    paper_id = _paper_cache_key(paper)
    cache_key = make_cache_key(
        "summary-v5-concise-paper-say",
        model or model_name("summary"),
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
        "Write concise plain-English paper summaries for builders. A summary "
        "answers only: what does this paper say? Be specific, avoid hype, "
        "avoid abstract-like phrasing, and do not invent details not supported "
        "by the title and abstract. Do not include conversational prefaces."
    )
    user_prompt = f"""
Paper title: {title}
Authors: {authors}
Abstract:
{abstract}

Write a concise summary in 2-4 sentences.

Rules:
- Do not use headings.
- Do not use bullet lists.
- Do not give builder recommendations or implementation advice.
- Do not restate the abstract.
- Do not mention benchmarks, datasets, libraries, or implementation details unless the abstract says them.
- If a key detail is missing, mention briefly that it should be verified in the full paper.
""".strip()

    summary = generate_text(system_prompt, user_prompt, token_budget=360, task="summary", model=model)
    SUMMARY_CACHE[cache_key] = summary
    cache_set("summaries", cache_key, summary)
    logger.info("Summary generation completed paper_id=%r", paper_id)
    return summary

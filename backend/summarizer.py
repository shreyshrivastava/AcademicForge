import logging

from backend.llm_text import PLAIN_STYLE
from backend.llm import generate_text
from backend.llm import model_name


logger = logging.getLogger(__name__)


def _paper_cache_key(paper):
    return paper.get("paper_id") or paper.get("url") or paper.get("link") or paper["title"]


def summarize_paper(paper, model=None, mode=None):
    """Generate a concise, useful research summary for one paper."""
    paper_id = _paper_cache_key(paper)
    logger.info("Summary generation started paper_id=%r", paper_id)
    title = paper["title"]
    authors = ", ".join(paper.get("authors", []))
    abstract = paper["abstract"]

    system_prompt = (
        "You explain research papers to AI engineers. Say what problem the paper tackles, "
        "what its main idea is, and what it means for someone building a system. "
        "Answer in English whatever the input language. " + PLAIN_STYLE
    )

    user_prompt = f"""TITLE: {title}
AUTHORS: {authors}

ABSTRACT:
{abstract}

TASK: Based on the abstract above, write a completely new 2-4 sentence summary in your own words. DO NOT copy the abstract."""

    summary = generate_text(system_prompt, user_prompt, token_budget=360, task="summary", model=model)
    logger.info("Summary generation completed paper_id=%r", paper_id)
    return summary

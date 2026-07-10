import logging

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

    system_prompt = """You are a specialized academic assistant. 
Your goal is to explain complex research to AI engineers and builders.
Write concise, plain-English summaries that skip the academic filler.
Focus strictly on: what is the problem, what is the core idea, and why it matters.
CRITICAL: You must respond exclusively in English, regardless of the input language."""

    user_prompt = f"""TITLE: {title}
AUTHORS: {authors}

ABSTRACT:
{abstract}

TASK: Based on the abstract above, write a completely new 2-4 sentence summary in your own words. DO NOT copy the abstract."""

    summary = generate_text(system_prompt, user_prompt, token_budget=360, task="summary", model=model)
    logger.info("Summary generation completed paper_id=%r", paper_id)
    return summary

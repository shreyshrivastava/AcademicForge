import logging
from backend.config import get_config
from backend.llm import generate_text, model_name
logger = logging.getLogger(__name__)


def _truncate(text, limit=900):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _paper_identity(paper):
    return paper.get("paper_id") or paper.get("url") or paper.get("link") or paper.get("title")


def build_paper_guidance_prompt(paper):
    authors = ", ".join(paper.get("authors", []))
    categories = ", ".join(paper.get("metadata", {}).get("categories", []))
    system_prompt = (
        "You are AcademicForge. Your job is to give short, practical guidance "
        "for how a builder can use one paper. Guidance answers only: how can I "
        "use this paper? Use only the paper title, abstract, and metadata. Do "
        "not include conversational prefaces and do not invent implementation "
        "details.\n"
        "CRITICAL: You must respond exclusively in English, regardless of the input language."
    )
    user_prompt = f"""
Evidence [1]
Paper title: {paper.get("title", "")}
Authors: {authors}
Date: {paper.get("date") or paper.get("published") or "unknown"}
Source: {paper.get("source", "arxiv")}
URL: {paper.get("url") or paper.get("link") or ""}
Categories: {categories}

Abstract:
{_truncate(paper.get("abstract", ""), 2500)}

Requirements:
- Return exactly three short paragraphs.
- Paragraph 1: why this paper matters.
- Paragraph 2: how to use it in a project.
- Paragraph 3: what to watch out for or verify.
- Do not use headings.
- Do not use bullet lists.
- Do not summarize the full paper.
- Only name benchmarks, datasets, models, algorithms, or libraries if they appear in the title, abstract, or metadata.
- Do not invent training steps, fine-tuning requirements, reward model designs, or evaluation names.
- If an implementation detail is missing, say briefly what to verify in the full paper.
- Never output "What this paper is about".
- Never output "Step-by-step reading plan".
- Never output "Difficulty level".
- Never output "Estimated learning path".
""".strip()
    return system_prompt, user_prompt


def generate_paper_guidance(paper, model=None, mode=None):
    """Generate practical guidance for understanding and applying one paper."""
    logger.info("Paper guidance generation started paper_id=%r", _paper_identity(paper))
    system_prompt, user_prompt = build_paper_guidance_prompt(paper)
    research_plan = generate_text(system_prompt, user_prompt, token_budget=1100, task="research_plan", model=model)
    logger.info("Paper guidance generation completed paper_id=%r", _paper_identity(paper))
    return research_plan




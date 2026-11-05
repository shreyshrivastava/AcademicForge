"""Balanced evidence-set selection from a ranked paper pool."""
from datetime import date
import logging
import re

logger = logging.getLogger(__name__)

FINAL_EVIDENCE_TARGET = 10
CORE_CATEGORY_QUOTAS = {
    "Foundational": 2,
    "Recent": 2,
    "Implementation Focused": 2,
    "Evaluation Focused": 1,
    "Alternative Approach": 1,
    "Contrarian View": 1,
}


def categorize_paper(paper: dict) -> str:
    """Assign a builder-facing evidence category using transparent heuristics."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    text = f"{title} {abstract}".lower()

    if _contains_any(text, ["survey", "review", "taxonomy", "overview", "systematic literature"]):
        return "Survey"
    if _contains_any(text, ["benchmark", "evaluation", "dataset", "metric", "leaderboard", "testbed"]):
        return "Evaluation Focused"
    if _contains_any(text, ["implementation", "framework", "system", "toolkit", "library", "pipeline", "deploy", "efficient", "lightweight"]):
        return "Implementation Focused"
    if _contains_any(text, ["limitation", "failure", "critique", "pitfall", "challenge", "rethink", "rethinking", "revisiting"]):
        return "Contrarian View"
    if _contains_any(text, ["alternative", "instead", "without", "non-", "contrastive", "different approach"]):
        return "Alternative Approach"
    if _paper_year(paper) >= date.today().year - 1:
        return "Recent"
    return "Foundational"


def select_evidence_set(
    ranked_papers: list[dict],
    target: int = FINAL_EVIDENCE_TARGET,
    preferred_categories: list[str] | None = None,
) -> list[dict]:
    """Select a balanced 8-10 paper evidence set from the reranked pool."""

    def in_rank_order(papers: list[dict]) -> list[dict]:
        return sorted(papers, key=lambda paper: paper.get("metadata", {}).get("selection_rank", 0))

    selected = []
    selected_ids = set()
    preferred_set = {category for category in (preferred_categories or []) if category}

    annotated = []
    for rank, paper in enumerate(ranked_papers, start=1):
        paper = dict(paper)
        category = categorize_paper(paper)
        paper["metadata"] = dict(paper.get("metadata", {}))
        paper["metadata"]["academicforge_category"] = category
        paper["metadata"]["selection_rank"] = rank
        annotated.append(paper)

    if preferred_set:
        for paper in annotated:
            if paper["metadata"]["academicforge_category"] in preferred_set:
                paper["metadata"]["category_focus_matched"] = True
                _append_unique(selected, selected_ids, paper)
                if len(selected) >= target:
                    return in_rank_order(selected)
        return in_rank_order(selected)

    for category, quota in CORE_CATEGORY_QUOTAS.items():
        for paper in [item for item in annotated if item["metadata"]["academicforge_category"] == category]:
            if len([item for item in selected if item["metadata"]["academicforge_category"] == category]) >= quota:
                break
            _append_unique(selected, selected_ids, paper)
            if len(selected) >= target:
                return in_rank_order(selected)

    for paper in annotated:
        _append_unique(selected, selected_ids, paper)
        if len(selected) >= target:
            break

    logger.info(
        "Evidence set selected ranked=%d selected=%d categories=%s",
        len(ranked_papers),
        len(selected),
        {
            category: len([paper for paper in selected if paper.get("metadata", {}).get("academicforge_category") == category])
            for category in sorted({paper.get("metadata", {}).get("academicforge_category") for paper in selected})
        },
    )
    return in_rank_order(selected)


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _paper_year(paper: dict) -> int:
    value = str(paper.get("published") or paper.get("date") or "")
    match = re.search(r"\b(19|20)\d{2}\b", value)
    if not match:
        return 0
    return int(match.group(0))


def _append_unique(selected: list[dict], selected_ids: set[str], paper: dict):
    paper_id = paper.get("paper_id") or paper.get("url") or paper.get("link") or paper.get("title")
    if paper_id in selected_ids:
        return
    selected_ids.add(paper_id)
    selected.append(paper)

from urllib.parse import urlparse
from datetime import date
import logging
import re

import arxiv

from backend.retrieval import hybrid_search, rerank_results
from backend.retrieval.models import result_to_paper


logger = logging.getLogger(__name__)

INITIAL_CANDIDATE_COUNT = 30
RERANK_POOL_SIZE = 30
FINAL_EVIDENCE_TARGET = 10

CORE_CATEGORY_QUOTAS = {
    "Foundational": 2,
    "Recent": 2,
    "Implementation Focused": 2,
    "Evaluation Focused": 1,
    "Alternative Approach": 1,
    "Contrarian View": 1,
}


def extract_arxiv_id(query: str) -> str | None:
    """Return an arXiv ID from a raw ID or arxiv.org URL."""
    value = query.strip()
    parsed = urlparse(value)
    if parsed.netloc.endswith("arxiv.org"):
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"abs", "pdf"}:
            return path_parts[1].removesuffix(".pdf")

    match = re.fullmatch(r"([a-z-]+/)?\d{7}|\d{4}\.\d{4,5}(v\d+)?", value)
    if match:
        return value.removesuffix(".pdf")

    return None


def search_arxiv_candidates(query: str, max_results: int = INITIAL_CANDIDATE_COUNT) -> tuple[list[dict], bool]:
    """Fetch arXiv candidates. Direct arXiv IDs bypass broad search."""
    arxiv_id = extract_arxiv_id(query)
    if arxiv_id:
        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
    else:
        search = arxiv.Search(query=query, max_results=max_results)

    papers = []
    client = arxiv.Client()
    for result in client.results(search):
        paper_url = result.pdf_url or result.entry_id
        papers.append(
            {
                "paper_id": result.entry_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "source": "arxiv",
                "url": paper_url,
                "link": paper_url,
                "published": result.published.date().isoformat(),
                "date": result.published.date().isoformat(),
                "metadata": {
                    "categories": list(result.categories or []),
                    "primary_category": result.primary_category,
                },
            }
        )

    logger.info("arXiv candidates fetched direct_id=%s count=%d", bool(arxiv_id), len(papers))
    return papers, bool(arxiv_id)


def rank_papers(query: str, papers: list[dict], top_k: int = RERANK_POOL_SIZE) -> list[dict]:
    """Rank candidate papers with hybrid retrieval and reranking."""
    retrieved = hybrid_search(query, papers, top_k=RERANK_POOL_SIZE)
    ranked = [
        result_to_paper(result)
        for result in rerank_results(query, retrieved, top_k=top_k)
    ]
    logger.info("Papers ranked pool_size=%d", len(ranked))
    return ranked


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
) -> list[dict]:
    """Select a balanced 8-10 paper evidence set from the reranked pool."""
    selected = []
    selected_ids = set()

    annotated = []
    for rank, paper in enumerate(ranked_papers, start=1):
        paper = dict(paper)
        category = categorize_paper(paper)
        paper["metadata"] = dict(paper.get("metadata", {}))
        paper["metadata"]["academicforge_category"] = category
        paper["metadata"]["selection_rank"] = rank
        annotated.append(paper)

    for category, quota in CORE_CATEGORY_QUOTAS.items():
        for paper in [item for item in annotated if item["metadata"]["academicforge_category"] == category]:
            if len([item for item in selected if item["metadata"]["academicforge_category"] == category]) >= quota:
                break
            _append_unique(selected, selected_ids, paper)
            if len(selected) >= target:
                return selected

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
    return selected


def retrieve_and_rank_papers(query: str) -> list[dict]:
    """End-to-end retrieval pipeline used by the API."""
    papers, direct_id = search_arxiv_candidates(query)
    if direct_id:
        return select_evidence_set(papers, target=1)
    ranked = rank_papers(query, papers)
    return select_evidence_set(ranked)


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

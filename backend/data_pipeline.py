from urllib.parse import urlparse
import logging
import re

import arxiv

from backend.retrieval import hybrid_search, rerank_results
from backend.retrieval.models import result_to_paper


logger = logging.getLogger(__name__)


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


def search_arxiv_candidates(query: str, max_results: int = 50) -> tuple[list[dict], bool]:
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


def rank_papers(query: str, papers: list[dict], top_k: int = 5) -> list[dict]:
    """Rank candidate papers with hybrid retrieval and reranking."""
    retrieved = hybrid_search(query, papers, top_k=20)
    return [
        result_to_paper(result)
        for result in rerank_results(query, retrieved, top_k=top_k)
    ]


def retrieve_and_rank_papers(query: str) -> list[dict]:
    """End-to-end retrieval pipeline used by the API."""
    papers, direct_id = search_arxiv_candidates(query)
    if direct_id:
        return papers
    return rank_papers(query, papers)

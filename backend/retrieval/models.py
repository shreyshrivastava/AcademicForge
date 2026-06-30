from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    source: str
    url: str
    published: str
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def paper_to_result(paper: dict, source: str = "arxiv") -> RetrievalResult:
    url = paper.get("url") or paper.get("link") or paper.get("entry_id") or ""
    title = paper.get("title", "")
    abstract = paper.get("abstract") or paper.get("summary") or ""
    paper_id = str(paper.get("paper_id") or paper.get("id") or url or title)
    return RetrievalResult(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=list(paper.get("authors", [])),
        source=paper.get("source", source),
        url=url,
        published=str(paper.get("published") or paper.get("date") or ""),
        bm25_rank=paper.get("bm25_rank"),
        dense_rank=paper.get("dense_rank"),
        rrf_score=float(paper.get("rrf_score", 0.0) or 0.0),
        metadata=dict(paper.get("metadata", {})),
    )


def result_to_paper(result: RetrievalResult) -> dict:
    return {
        "paper_id": result.paper_id,
        "title": result.title,
        "authors": result.authors,
        "abstract": result.abstract,
        "source": result.source,
        "url": result.url,
        "link": result.url,
        "published": result.published,
        "date": result.published,
        "bm25_rank": result.bm25_rank,
        "dense_rank": result.dense_rank,
        "rrf_score": result.rrf_score,
        "metadata": result.metadata,
    }

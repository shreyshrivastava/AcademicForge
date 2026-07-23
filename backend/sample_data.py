"""Synthetic paper records used for offline demos, tests, and benchmarks."""

from copy import deepcopy


SAMPLE_PAPERS: list[dict] = [
    {
        "paper_id": "demo-rag-001",
        "title": "Grounded Retrieval for Reducing Hallucinations in Language Models",
        "authors": ["A. Patel", "M. Chen"],
        "abstract": (
            "This synthetic demo paper studies retrieval augmented generation systems "
            "that verify generated claims against retrieved passages. It discusses "
            "claim extraction, evidence matching, factuality scoring, and evaluation "
            "metrics for reducing hallucinated answers in LLM applications."
        ),
        "source": "synthetic_demo",
        "url": "https://example.com/academicforge/demo-rag-001",
        "link": "https://example.com/academicforge/demo-rag-001",
        "published": "2025-02-14",
        "date": "2025-02-14",
        "metadata": {
            "fields_of_study": ["Computer Science"],
            "keywords": ["rag", "hallucination", "factuality", "evaluation"],
            "citation_count": None,
            "demo_record": True,
        },
    },
    {
        "paper_id": "demo-rag-002",
        "title": "Benchmarking Factuality in Retrieval-Augmented Question Answering",
        "authors": ["S. Rivera"],
        "abstract": (
            "This synthetic demo paper describes a benchmark for measuring answer "
            "grounding in retrieval augmented question answering. It compares exact "
            "match, citation precision, unsupported-claim rate, and retrieval recall "
            "under deterministic evaluation settings."
        ),
        "source": "synthetic_demo",
        "url": "https://example.com/academicforge/demo-rag-002",
        "link": "https://example.com/academicforge/demo-rag-002",
        "published": "2024-11-02",
        "date": "2024-11-02",
        "metadata": {
            "fields_of_study": ["Computer Science"],
            "keywords": ["benchmark", "factuality", "retrieval", "qa"],
            "citation_count": None,
            "demo_record": True,
        },
    },
    {
        "paper_id": "demo-rag-003",
        "title": "Lightweight RAG Pipelines for Production Assistants",
        "authors": ["K. Morgan", "D. Singh"],
        "abstract": (
            "This synthetic demo paper presents an implementation-focused retrieval "
            "pipeline with lexical search, dense embeddings, rank fusion, prompt "
            "assembly, and operational checks for latency, fallback behavior, and "
            "source attribution."
        ),
        "source": "synthetic_demo",
        "url": "https://example.com/academicforge/demo-rag-003",
        "link": "https://example.com/academicforge/demo-rag-003",
        "published": "2025-06-20",
        "date": "2025-06-20",
        "metadata": {
            "fields_of_study": ["Computer Science"],
            "keywords": ["implementation", "rag", "latency", "deployment"],
            "citation_count": None,
            "demo_record": True,
        },
    },
    {
        "paper_id": "demo-rag-004",
        "title": "Failure Modes of Retrieval Grounded Generation",
        "authors": ["J. Lee"],
        "abstract": (
            "This synthetic demo paper reviews limitations of retrieval grounded "
            "generation including stale sources, ambiguous evidence, citation drift, "
            "query mismatch, and overconfident synthesis when retrieved context is "
            "incomplete."
        ),
        "source": "synthetic_demo",
        "url": "https://example.com/academicforge/demo-rag-004",
        "link": "https://example.com/academicforge/demo-rag-004",
        "published": "2023-09-18",
        "date": "2023-09-18",
        "metadata": {
            "fields_of_study": ["Computer Science"],
            "keywords": ["limitations", "rag", "retrieval", "hallucination"],
            "citation_count": None,
            "demo_record": True,
        },
    },
]


def get_sample_papers(limit: int | None = None) -> list[dict]:
    """Return isolated sample-paper copies so callers can safely add metadata."""
    papers = deepcopy(SAMPLE_PAPERS)
    return papers[:limit] if limit else papers

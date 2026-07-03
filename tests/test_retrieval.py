import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.retrieval.bm25 import bm25_search
from backend.retrieval.dense import dense_search
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.rrf import reciprocal_rank_fusion
from backend.data_pipeline import categorize_paper, select_evidence_set


PAPERS = [
    {
        "paper_id": "paper-a",
        "title": "Transformer Retrieval for Multilingual Fact Checking",
        "abstract": "A dense retrieval system for claim verification across languages.",
        "authors": ["Ada"],
        "source": "test",
        "url": "https://example.com/a",
        "published": "2025-01-01",
        "metadata": {"categories": ["cs.CL"], "keywords": ["fact checking", "retrieval"]},
    },
    {
        "paper_id": "paper-b",
        "title": "Graph Neural Networks for Molecules",
        "abstract": "Message passing models for molecular property prediction.",
        "authors": ["Bert"],
        "source": "test",
        "url": "https://example.com/b",
        "published": "2025-01-02",
    },
    {
        "paper_id": "paper-c",
        "title": "Cross Lingual Claim Verification",
        "abstract": "Semantic search and evidence retrieval for multilingual misinformation detection.",
        "authors": ["Cy"],
        "source": "test",
        "url": "https://example.com/c",
        "published": "2025-01-03",
    },
]


def test_bm25_returns_results():
    results = bm25_search("multilingual fact checking retrieval", PAPERS)
    assert results
    assert results[0].bm25_rank == 1


def test_dense_returns_results():
    results = dense_search("semantic multilingual evidence retrieval", PAPERS)
    assert results
    assert results[0].dense_rank == 1


def test_rrf_combines_duplicates_and_promotes_overlap():
    bm25_results = bm25_search("multilingual fact checking retrieval", PAPERS)
    dense_results = dense_search("multilingual fact checking retrieval", PAPERS)
    fused = reciprocal_rank_fusion([bm25_results, dense_results], top_k=3)
    ids = [result.paper_id for result in fused]

    assert len(ids) == len(set(ids))
    assert fused[0].paper_id in {"paper-a", "paper-c"}
    assert fused[0].rrf_score > 0
    assert fused[0].bm25_rank is not None
    assert fused[0].dense_rank is not None


def test_hybrid_includes_rrf_score():
    results = hybrid_search("multilingual fact checking retrieval", PAPERS, top_k=2)
    assert results
    assert all(result.rrf_score > 0 for result in results)


def test_categorize_paper_uses_builder_facing_labels():
    assert categorize_paper({"title": "A Survey of RAG", "abstract": "", "date": "2024-01-01"}) == "Survey"
    assert categorize_paper({"title": "Open Benchmark for Hallucination Evaluation", "abstract": "", "date": "2024-01-01"}) == "Evaluation Focused"
    assert categorize_paper({"title": "A Lightweight Deployment Framework", "abstract": "", "date": "2024-01-01"}) == "Implementation Focused"
    assert categorize_paper({"title": "Limitations and Failure Modes of RAG", "abstract": "", "date": "2024-01-01"}) == "Contrarian View"


def test_select_evidence_set_balances_categories_and_caps_size():
    ranked = []
    templates = [
        ("foundational", "Classic Retrieval Transformer", "2021-01-01"),
        ("recent", "New Method for RAG", "2026-01-01"),
        ("implementation", "Lightweight Deployment Framework", "2025-01-01"),
        ("evaluation", "Benchmark for Hallucination Evaluation", "2025-01-01"),
        ("alternative", "Alternative Approach without Dense Retrieval", "2025-01-01"),
        ("contrarian", "Limitations and Failure Modes of RAG", "2025-01-01"),
    ]
    for index in range(18):
        _, title, date = templates[index % len(templates)]
        ranked.append(
            {
                "paper_id": f"paper-{index}",
                "title": f"{title} {index}",
                "abstract": "A test abstract.",
                "authors": ["Ada"],
                "source": "test",
                "url": f"https://example.com/{index}",
                "link": f"https://example.com/{index}",
                "date": date,
                "published": date,
                "metadata": {},
            }
        )

    selected = select_evidence_set(ranked, target=10)
    categories = {paper["metadata"]["academicforge_category"] for paper in selected}

    assert len(selected) == 10
    assert "Implementation Focused" in categories
    assert "Evaluation Focused" in categories
    assert "Contrarian View" in categories
    assert "Alternative Approach" in categories
    assert all("selection_rank" in paper["metadata"] for paper in selected)


if __name__ == "__main__":
    test_bm25_returns_results()
    test_dense_returns_results()
    test_rrf_combines_duplicates_and_promotes_overlap()
    test_hybrid_includes_rrf_score()
    test_categorize_paper_uses_builder_facing_labels()
    test_select_evidence_set_balances_categories_and_caps_size()
    print("retrieval tests passed")

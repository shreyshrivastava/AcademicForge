import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.retrieval.bm25 import bm25_search
from backend.retrieval.dense import dense_search
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.rrf import reciprocal_rank_fusion
import backend.data_pipeline as data_pipeline
from backend.data_pipeline import categorize_paper, filter_relevant_papers, infer_query_domain, retrieve_and_rank_papers, select_evidence_set


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

DOMAIN_PAPERS = [
    {
        "paper_id": "health-1",
        "title": "Exercise and dietary interventions for obesity and fat loss",
        "abstract": "Clinical evidence on reducing body fat, obesity, weight, metabolism, nutrition, and physical activity.",
        "authors": ["Ada"],
        "source": "semantic_scholar",
        "url": "https://example.com/health-1",
        "published": "2024",
        "metadata": {"fields_of_study": ["Medicine"]},
    },
    {
        "paper_id": "health-2",
        "title": "Metabolic effects of calorie restriction on adipose tissue",
        "abstract": "Nutrition and diet interventions affect lipid metabolism, adipose tissue, overweight adults, and healthcare outcomes.",
        "authors": ["Ben"],
        "source": "semantic_scholar",
        "url": "https://example.com/health-2",
        "published": "2023",
        "metadata": {"fields_of_study": ["Medicine", "Biology"]},
    },
    {
        "paper_id": "health-3",
        "title": "Weight management through physical activity and nutrition",
        "abstract": "A health study of obesity treatment, exercise programs, dietary adherence, body mass, and fat reduction.",
        "authors": ["Cy"],
        "source": "semantic_scholar",
        "url": "https://example.com/health-3",
        "published": "2022",
        "metadata": {"fields_of_study": ["Medicine"]},
    },
    {
        "paper_id": "ai-1",
        "title": "Reducing LLM hallucinations in retrieval augmented generation",
        "abstract": "A factuality method for language models using retrieval evidence, answer verification, and RAG evaluation.",
        "authors": ["Dee"],
        "source": "semantic_scholar",
        "url": "https://example.com/ai-1",
        "published": "2025",
        "metadata": {"fields_of_study": ["Computer Science"]},
    },
    {
        "paper_id": "ai-2",
        "title": "Hallucination detection for large language models",
        "abstract": "NLP research on factuality, generation errors, transformer models, and retrieval grounded language model answers.",
        "authors": ["Eve"],
        "source": "semantic_scholar",
        "url": "https://example.com/ai-2",
        "published": "2024",
        "metadata": {"fields_of_study": ["Computer Science"]},
    },
    {
        "paper_id": "ai-3",
        "title": "Benchmarks for RAG factuality and hallucination",
        "abstract": "Evaluation datasets for LLM hallucinations, retrieval systems, model answers, and NLP factuality metrics.",
        "authors": ["Fox"],
        "source": "semantic_scholar",
        "url": "https://example.com/ai-3",
        "published": "2024",
        "metadata": {"fields_of_study": ["Computer Science"]},
    },
    {
        "paper_id": "software-1",
        "title": "AI for software testing and automated test generation",
        "abstract": "Software engineering methods for testing code, bug detection, program verification, and developer workflows.",
        "authors": ["Gus"],
        "source": "semantic_scholar",
        "url": "https://example.com/software-1",
        "published": "2025",
        "metadata": {"fields_of_study": ["Computer Science"]},
    },
    {
        "paper_id": "software-2",
        "title": "Large language models for unit tests in software engineering",
        "abstract": "Developer tools for code testing, test suite generation, debugging, and program repair.",
        "authors": ["Hal"],
        "source": "semantic_scholar",
        "url": "https://example.com/software-2",
        "published": "2024",
        "metadata": {"fields_of_study": ["Computer Science"]},
    },
    {
        "paper_id": "software-3",
        "title": "Neural bug detection for software verification",
        "abstract": "Testing software systems with AI models, code analysis, bug localization, and verification benchmarks.",
        "authors": ["Ivy"],
        "source": "semantic_scholar",
        "url": "https://example.com/software-3",
        "published": "2023",
        "metadata": {"fields_of_study": ["Computer Science"]},
    },
    {
        "paper_id": "energy-1",
        "title": "Solar energy forecasting with deep learning",
        "abstract": "Renewable energy forecasting for photovoltaic generation, grid planning, and solar power prediction.",
        "authors": ["Jay"],
        "source": "semantic_scholar",
        "url": "https://example.com/energy-1",
        "published": "2024",
        "metadata": {"fields_of_study": ["Engineering"]},
    },
    {
        "paper_id": "energy-2",
        "title": "Probabilistic forecasting for photovoltaic energy systems",
        "abstract": "Solar generation, renewable power forecasting, weather signals, and grid energy management.",
        "authors": ["Kay"],
        "source": "semantic_scholar",
        "url": "https://example.com/energy-2",
        "published": "2023",
        "metadata": {"fields_of_study": ["Engineering"]},
    },
    {
        "paper_id": "energy-3",
        "title": "Short-term solar irradiance prediction for renewable grids",
        "abstract": "Forecasting solar irradiance and photovoltaic output for energy systems and grid operations.",
        "authors": ["Lee"],
        "source": "semantic_scholar",
        "url": "https://example.com/energy-3",
        "published": "2022",
        "metadata": {"fields_of_study": ["Engineering"]},
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


def test_select_evidence_set_prefers_requested_categories():
    ranked = []
    for index in range(14):
        is_implementation = index < 10
        ranked.append(
            {
                "paper_id": f"paper-{index}",
                "title": (
                    f"Lightweight Deployment Framework {index}"
                    if is_implementation
                    else f"Classic Retrieval Transformer {index}"
                ),
                "abstract": "A test abstract.",
                "authors": ["Ada"],
                "source": "test",
                "url": f"https://example.com/{index}",
                "link": f"https://example.com/{index}",
                "date": "2025-01-01" if is_implementation else "2021-01-01",
                "published": "2025-01-01" if is_implementation else "2021-01-01",
                "metadata": {},
            }
        )

    selected = select_evidence_set(
        ranked,
        target=10,
        preferred_categories=["Implementation Focused"],
    )

    assert len(selected) == 10
    assert all(paper["metadata"]["academicforge_category"] == "Implementation Focused" for paper in selected)
    assert all(paper["metadata"]["category_focus_matched"] for paper in selected)


def test_filter_relevant_papers_removes_domain_mismatch_for_fat_loss():
    relevant = filter_relevant_papers("how to reduce fat", DOMAIN_PAPERS)
    titles = " ".join(paper["title"].lower() for paper in relevant)

    assert infer_query_domain("how to reduce fat") == "health"
    assert len(relevant) >= 3
    assert any(term in titles for term in ["obesity", "metabolic", "weight", "exercise"])
    assert "software" not in titles
    assert "testing" not in titles


def test_retrieve_and_rank_papers_uses_live_relevant_candidates_across_domains():
    original_search = data_pipeline.search_live_candidates

    def fake_search(query, max_results=data_pipeline.INITIAL_CANDIDATE_COUNT):
        return list(DOMAIN_PAPERS), False

    data_pipeline.search_live_candidates = fake_search
    try:
        expectations = [
            ("how to reduce fat", "health", ["obesity", "metabolic", "weight", "exercise"], ["software", "testing"]),
            ("how to reduce LLM hallucinations", "ai", ["hallucination", "rag", "factuality"], ["obesity", "solar"]),
            ("AI for software testing", "software", ["software", "testing", "bug"], ["obesity", "solar"]),
            ("solar energy forecasting", "energy", ["solar", "forecasting", "photovoltaic"], ["software", "obesity"]),
        ]
        for query, expected_domain, required_terms, banned_terms in expectations:
            selected = retrieve_and_rank_papers(query)
            selected_text = " ".join(
                f"{paper['title']} {paper['abstract']}".lower()
                for paper in selected
            )
            assert selected, query
            assert infer_query_domain(query) == expected_domain
            assert any(term.lower() in selected_text for term in required_terms), query
            assert not any(term.lower() in selected_text for term in banned_terms), query
            assert all(paper["metadata"]["relevance_score"] >= data_pipeline.RELEVANCE_THRESHOLD for paper in selected)
    finally:
        data_pipeline.search_live_candidates = original_search


def test_rerank_results_cross_encoder():
    from backend.retrieval.reranker import rerank_results
    from backend.retrieval.models import RetrievalResult

    results = [
        RetrievalResult(
            paper_id="1",
            title="Transformer Retrieval for Multilingual Fact Checking",
            abstract="A dense retrieval system for claim verification across languages.",
            authors=["Ada"],
            source="test",
            url="https://example.com/a",
            published="2025-01-01",
        ),
        RetrievalResult(
            paper_id="2",
            title="Graph Neural Networks for Molecules",
            abstract="Message passing models for molecular property prediction.",
            authors=["Bert"],
            source="test",
            url="https://example.com/b",
            published="2025-01-02",
        ),
    ]

    # Rerank with query related to fact checking
    reranked = rerank_results("fact checking claims verification", results, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].paper_id == "1"

    # Test empty list
    assert rerank_results("test", []) == []


if __name__ == "__main__":
    test_bm25_returns_results()
    test_dense_returns_results()
    test_rrf_combines_duplicates_and_promotes_overlap()
    test_hybrid_includes_rrf_score()
    test_categorize_paper_uses_builder_facing_labels()
    test_select_evidence_set_balances_categories_and_caps_size()
    test_select_evidence_set_prefers_requested_categories()
    test_filter_relevant_papers_removes_domain_mismatch_for_fat_loss()
    test_retrieve_and_rank_papers_uses_live_relevant_candidates_across_domains()
    test_rerank_results_cross_encoder()
    print("retrieval tests passed")

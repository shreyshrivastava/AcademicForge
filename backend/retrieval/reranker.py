from backend.retrieval.models import RetrievalResult


def rerank_results(query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
    # TODO: Add cross-encoder reranking here when the MVP retrieval flow is validated.
    return results[:top_k]

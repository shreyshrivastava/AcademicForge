from sentence_transformers import CrossEncoder
from backend.retrieval.models import RetrievalResult

# Lazy-loaded global cross-encoder instance
_MODEL = None


def get_reranker_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder("BAAI/bge-reranker-base")
    return _MODEL


def rerank_results(query: str, results: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
    if not results:
        return []

    try:
        model = get_reranker_model()
        # Pair the query with each document's title and abstract
        pairs = [(query, f"{r.title}\n{r.abstract}") for r in results]
        scores = model.predict(pairs)

        # Sort results based on score descending
        scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored][:top_k]
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Cross-encoder reranking failed: {e}. Falling back to default order.")
        return results[:top_k]

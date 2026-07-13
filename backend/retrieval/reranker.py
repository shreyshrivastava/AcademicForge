import logging

from backend.retrieval.device import select_retrieval_device
from backend.retrieval.models import RetrievalResult

# Lazy-loaded global cross-encoder instance
_MODEL = None
logger = logging.getLogger(__name__)


def get_reranker_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder

        device = select_retrieval_device()
        try:
            logger.info("Loading cross-encoder reranker BAAI/bge-reranker-base on %s", device)
            _MODEL = CrossEncoder("BAAI/bge-reranker-base", device=device)
        except Exception as exc:
            if device == "cpu":
                raise
            logger.warning("Cross-encoder GPU load failed on %s: %s. Falling back to CPU.", device, exc)
            _MODEL = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
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

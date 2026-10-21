import logging
from threading import Lock

from backend.retrieval.device import select_retrieval_device
from backend.retrieval.models import RetrievalResult

# Lazy-loaded global cross-encoder instance
_MODEL = None
_MODEL_LOCK = Lock()
logger = logging.getLogger(__name__)


def _get_reranker_model():
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


def get_reranker_model():
    with _MODEL_LOCK:
        return _get_reranker_model()


def score_pairs_length_sorted(model, pairs: list[tuple[str, str]], batch_size: int = 32):
    """Score pairs in shorter batches first, then restore their original order.

    Sorting only changes padding efficiency.  The returned score at each index
    still belongs to the corresponding input pair, so ranking semantics remain
    unchanged.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    ordered = sorted(enumerate(pairs), key=lambda item: len(item[1][0]) + len(item[1][1]))
    ordered_scores = model.predict([pair for _, pair in ordered], batch_size=batch_size)
    if len(ordered_scores) != len(pairs):
        raise ValueError("Cross-encoder returned a score count that does not match the input pairs")
    scores = [None] * len(pairs)
    for (original_index, _), score in zip(ordered, ordered_scores):
        scores[original_index] = score
    return scores


def rerank_results(
    query: str,
    results: list[RetrievalResult],
    top_k: int = 5,
    batch_size: int = 32,
) -> list[RetrievalResult]:
    if not results:
        return []

    try:
        model = get_reranker_model()
        # Pair the query with each document's title and abstract
        pairs = [(query, f"{r.title}\n{r.abstract}") for r in results]
        scores = score_pairs_length_sorted(model, pairs, batch_size=batch_size)

        # Sort results based on score descending
        scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        for result, _ in scored:
            result.metadata["reranker_backend"] = "BAAI/bge-reranker-base"
        return [r for r, _ in scored][:top_k]
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Cross-encoder reranking failed: {e}. Falling back to default order.")
        for result in results:
            result.metadata["reranker_backend"] = "original_order"
        return results[:top_k]

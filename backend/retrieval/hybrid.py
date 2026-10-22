import logging

from backend.retrieval.bm25 import BM25Index, stem_and_remove_stopwords
from backend.retrieval.dense import dense_search
from backend.retrieval.rrf import reciprocal_rank_fusion


logger = logging.getLogger(__name__)


def hybrid_search(query: str, papers: list[dict], top_k: int = 20):
    from backend.retrieval.bm25 import bm25_search
    bm25_results = bm25_search(query, papers, top_k=50)
    dense_results = dense_search(query, papers, top_k=50)
    fused_results = reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)
    logger.info(
        "Hybrid search completed candidates=%d bm25=%d dense=%d fused=%d",
        len(papers),
        len(bm25_results),
        len(dense_results),
        len(fused_results),
    )
    return fused_results


def score_fusion_search(query: str, papers: list[dict], top_k: int = 20, dense_alpha: float = 0.9):
    """Fuse normalized BGE and stemmed-BM25 scores with the frozen alpha 0.9."""
    if not 0 <= dense_alpha <= 1:
        raise ValueError("dense_alpha must be between zero and one")
    dense_results = dense_search(query, papers, top_k=len(papers))
    lexical_results = BM25Index(papers, analyzer=stem_and_remove_stopwords).search(query, top_k=len(papers))
    if not dense_results:
        return lexical_results[:top_k]
    dense_values = [result.metadata.get("dense_score", 0.0) for result in dense_results]
    lexical_by_id = {result.paper_id: result.metadata.get("bm25_score", 0.0) for result in lexical_results}
    dense_low, dense_high = min(dense_values), max(dense_values)
    lexical_values = list(lexical_by_id.values()) or [0.0]
    lexical_low, lexical_high = min(lexical_values), max(lexical_values)
    dense_range = max(dense_high - dense_low, 1e-12)
    lexical_range = max(lexical_high - lexical_low, 1e-12)
    scored = []
    for result in dense_results:
        dense_score = (result.metadata.get("dense_score", 0.0) - dense_low) / dense_range
        lexical_score = (lexical_by_id.get(result.paper_id, 0.0) - lexical_low) / lexical_range
        result.metadata["score_fusion_alpha"] = dense_alpha
        result.metadata["score_fusion_score"] = dense_alpha * dense_score + (1 - dense_alpha) * lexical_score
        scored.append(result)
    ranked = sorted(scored, key=lambda result: result.metadata["score_fusion_score"], reverse=True)[:top_k]
    logger.info("Score fusion completed candidates=%d dense_alpha=%.1f", len(papers), dense_alpha)
    return ranked

import logging

from backend.retrieval.bm25 import bm25_search
from backend.retrieval.dense import dense_search
from backend.retrieval.rrf import reciprocal_rank_fusion


logger = logging.getLogger(__name__)


def hybrid_search(query: str, papers: list[dict], top_k: int = 20):
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

from backend.retrieval.bm25 import bm25_search
from backend.retrieval.dense import dense_search
from backend.retrieval.rrf import reciprocal_rank_fusion


def hybrid_search(query: str, papers: list[dict], top_k: int = 20):
    bm25_results = bm25_search(query, papers, top_k=50)
    dense_results = dense_search(query, papers, top_k=50)
    return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)

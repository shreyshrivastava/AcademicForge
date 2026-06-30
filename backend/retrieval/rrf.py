from copy import deepcopy

from backend.retrieval.models import RetrievalResult


def _merge_result(existing: RetrievalResult, incoming: RetrievalResult) -> None:
    if incoming.bm25_rank is not None:
        existing.bm25_rank = (
            incoming.bm25_rank
            if existing.bm25_rank is None
            else min(existing.bm25_rank, incoming.bm25_rank)
        )
    if incoming.dense_rank is not None:
        existing.dense_rank = (
            incoming.dense_rank
            if existing.dense_rank is None
            else min(existing.dense_rank, incoming.dense_rank)
        )
    existing.metadata.update(incoming.metadata)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int = 20,
) -> list[RetrievalResult]:
    fused: dict[str, RetrievalResult] = {}

    for ranked_list in ranked_lists:
        seen_in_list = set()
        for rank, result in enumerate(ranked_list, start=1):
            if result.paper_id in seen_in_list:
                continue
            seen_in_list.add(result.paper_id)

            if result.paper_id not in fused:
                fused[result.paper_id] = deepcopy(result)
                fused[result.paper_id].rrf_score = 0.0
            else:
                _merge_result(fused[result.paper_id], result)

            fused[result.paper_id].rrf_score += 1 / (k + rank)

    return sorted(
        fused.values(),
        key=lambda result: (
            result.rrf_score,
            -(result.bm25_rank or 1_000_000),
            -(result.dense_rank or 1_000_000),
        ),
        reverse=True,
    )[:top_k]

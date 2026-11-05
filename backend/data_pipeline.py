"""Search orchestration: live sources, ranking, relevance filtering and evidence selection."""
import logging
from types import SimpleNamespace

from backend.retrieval import hybrid_search, rerank_results, score_fusion_search
from backend.retrieval.bm25 import bm25_search
from backend.retrieval.dense import dense_search
from backend.retrieval.models import result_to_paper
from backend.jev import enrich_papers_with_jev, rerank_papers_with_jev
from backend.config import get_config
from backend.paper_search import _dedupe_papers, extract_arxiv_id, search_live_candidates
from backend.relevance import _content_tokens, filter_relevant_papers, infer_query_domain
from backend.evidence import FINAL_EVIDENCE_TARGET, select_evidence_set
from backend.search_cache import cache_key, get_cached, store
from backend.errors import SourceUnavailable
from backend.search_context import contextualize_query, matches_field, field_kind
from backend.sources import begin_search, diagnostics, search_diagnostics


logger = logging.getLogger(__name__)

EXPANDED_CANDIDATE_COUNT = 100
RERANK_POOL_SIZE = 30
MIN_RELEVANT_PAPERS = 3


def rewrite_search_query(query: str) -> str:
    """Convert casual user phrasing into an academic source query when useful."""
    if extract_arxiv_id(query):
        return query.strip()
    query_domain = infer_query_domain(query)
    terms = set(_content_tokens(query))
    if query_domain == "health" and "fat" in terms and ({"reduce", "reducing", "reduction", "loss"} & terms):
        return "fat loss obesity weight loss diet exercise nutrition metabolism"
    return query.strip()


def rank_papers(query: str, papers: list[dict], top_k: int = RERANK_POOL_SIZE) -> list[dict]:
    """Rank papers using the evidence-backed default or an explicit opt-in mode."""
    mode = get_config().retrieval_mode
    if mode == "dense":
        ranked = [result_to_paper(result) for result in dense_search(query, papers, top_k=top_k)]
    elif mode == "score_fusion":
        ranked = [result_to_paper(result) for result in score_fusion_search(query, papers, top_k=top_k, dense_alpha=.9)]
    else:
        retrieved = hybrid_search(query, papers, top_k=RERANK_POOL_SIZE)
        if mode == "hybrid_rerank":
            retrieved = rerank_results(query, retrieved, top_k=top_k)
        else:
            retrieved = retrieved[:top_k]
        ranked = [result_to_paper(result) for result in retrieved]
    ranked = rerank_papers_with_jev(query, ranked)
    logger.info("Papers ranked mode=%s pool_size=%d", mode, len(ranked))
    return ranked


def not_enough_relevant_message() -> str:
    return "Not enough relevant papers found. Try a more specific academic query."


def _filter_for_field(query, papers, domain):
    if field_kind(domain) != 'all':
        # Explicit user intent takes precedence over the old automatic domain guess.
        # The field predicate has already filtered candidates; hybrid ranking handles topic relevance.
        return papers
    return filter_relevant_papers(query, papers)


def retrieve_and_rank_papers(query: str, preferred_categories: list[str] | None = None, domain: str = "", *, lightweight=False) -> list[dict]:
    """End-to-end retrieval pipeline used by the API."""
    begin_search()
    original_query = query.strip()
    # The standalone profile never reads generation credentials or initializes models.
    config = SimpleNamespace(search_mode="standalone_bm25") if lightweight else get_config()
    key = cache_key(original_query, preferred_categories, config.search_mode, domain)
    intent_query = contextualize_query(original_query, domain) if not extract_arxiv_id(original_query) else original_query
    cached = get_cached(key)
    if cached is not None:
        logger.info("Search cache hit result_count=%d", len(cached))
        search_diagnostics.set({"sources": {"cache": "fresh"}, "warnings": []})
        return cached

    search_query = rewrite_search_query(intent_query)
    logger.info(
        "Retrieval query original_query=%r rewritten_query=%r source_query=%r",
        original_query,
        search_query,
        search_query,
    )
    candidates, direct_id = search_live_candidates(search_query)
    if direct_id:
        candidates = [paper for paper in candidates if matches_field(paper, original_query, domain)]
        selected = select_evidence_set(candidates, target=1)
        if not lightweight:
            selected = enrich_papers_with_jev(original_query, selected)
        if not diagnostics()["warnings"]:
            store(key, selected)
        return selected
    candidates = [paper for paper in candidates if matches_field(paper, original_query, domain)]
    papers = _filter_for_field(intent_query, candidates, domain)
    if len(papers) < FINAL_EVIDENCE_TARGET and not diagnostics()["warnings"]:
        logger.info(
            "Expanding source search original_query=%r relevant_count=%d target=%d expanded_candidates=%d",
            original_query,
            len(papers),
            FINAL_EVIDENCE_TARGET,
            EXPANDED_CANDIDATE_COUNT,
        )
        try:
            expanded_candidates, _ = search_live_candidates(
                search_query, max_results=EXPANDED_CANDIDATE_COUNT,
            )
        except SourceUnavailable:
            expanded_candidates = []  # Keep evidence from the first successful fetch.
        candidates = _dedupe_papers([*candidates, *expanded_candidates])
        candidates = [paper for paper in candidates if matches_field(paper, original_query, domain)]
        papers = _filter_for_field(intent_query, candidates, domain)
    if not papers:
        logger.warning(
            "Not enough relevant papers original_query=%r relevant_count=%d",
            original_query,
            len(papers),
        )
        return []
    if lightweight:
        ranked = [result_to_paper(result) for result in bm25_search(intent_query, papers, top_k=RERANK_POOL_SIZE)]
        for paper in ranked:
            paper["metadata"].update({"ranking_backend": "bm25", "dense_backend": "disabled",
                                      "reranker_backend": "disabled"})
    else:
        ranked = rank_papers(intent_query, papers)
    selected = select_evidence_set(ranked, preferred_categories=preferred_categories)
    _log_final_selection(original_query, selected)
    if not diagnostics()["warnings"]:
        store(key, selected)
    return selected


def _log_final_selection(query: str, selected: list[dict]):
    for index, paper in enumerate(selected, start=1):
        metadata = paper.get("metadata", {}) or {}
        logger.info(
            "Final selected query=%r rank=%d title=%r score=%s source=%s source_category=%s abstract=%r",
            query,
            index,
            paper.get("title"),
            metadata.get("relevance_score"),
            paper.get("source"),
            metadata.get("fields_of_study") or metadata.get("categories") or metadata.get("primary_category"),
            (paper.get("abstract") or "")[:500],
        )

"""Optional Jev decision layer for structured paper signals.

Jev is used only for typed decisions. Text generation remains in the existing
MLX/Transformers path, and search continues to work when Jev is disabled.
"""

import json
import logging
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256

import requests

from backend.config import AppConfig, get_config

logger = logging.getLogger(__name__)
_JEV_CACHE_MAX_SIZE = 64
_JEV_CACHE: OrderedDict[str, list[dict]] = OrderedDict()

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
PAPER_TYPES = {
    "method": "Introduces a new algorithm, model, or technical method.",
    "benchmark": "Introduces a benchmark, dataset, evaluation, or measurement framework.",
    "theory": "Primarily presents theory, analysis, or formal understanding.",
    "application": "Applies existing methods to a practical domain or product problem.",
    "survey": "Surveys, organizes, or reviews prior research.",
    "other": "Does not clearly fit the other paper types.",
}


def jev_enabled(config: AppConfig | None = None) -> bool:
    config = config or get_config()
    local_endpoint = config.jev_endpoint.startswith(("http://127.0.0.1", "http://localhost"))
    return bool(config.search_mode == "quality" and config.jev_enabled and (config.jev_api_key or local_endpoint))


def _paper_state(query: str, papers: list[dict]) -> str:
    compact = []
    for index, paper in enumerate(papers):
        compact.append(
            {
                "index": index,
                "title": paper.get("title", ""),
                "abstract": (paper.get("abstract") or paper.get("summary") or "")[:600],
            }
        )
    return json.dumps({"query": query, "papers": compact}, ensure_ascii=True)


def _cache_key(query: str, papers: list[dict], config: AppConfig) -> str:
    """Build a content-aware key so changed paper data cannot reuse stale scores."""
    state = {
        "query": query,
        "model": config.jev_model,
        "include_details": config.jev_include_details,
        "papers": [
            {
                "paper_id": paper.get("paper_id") or paper.get("id"),
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract") or paper.get("summary") or "",
                "metadata": paper.get("metadata", {}),
            }
            for paper in papers[: config.jev_max_papers]
        ],
    }
    serialized = json.dumps(state, sort_keys=True, ensure_ascii=True, default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def _cached_result(key: str) -> list[dict] | None:
    result = _JEV_CACHE.get(key)
    if result is None:
        return None
    _JEV_CACHE.move_to_end(key)
    logger.info("Jev cache hit")
    return deepcopy(result)


def _store_cached_result(key: str, result: list[dict]) -> None:
    _JEV_CACHE[key] = deepcopy(result)
    _JEV_CACHE.move_to_end(key)
    while len(_JEV_CACHE) > _JEV_CACHE_MAX_SIZE:
        _JEV_CACHE.popitem(last=False)


def _questions(paper_count: int, include_details: bool = False) -> dict:
    questions = {
        "paper_relevance": {
            "type": "choice",
            "instructions": "Which paper is most relevant to the research query?",
            "criteria": {
                f"paper_{index}": f"Paper {index}: the title and abstract shown in the state."
                for index in range(paper_count)
            },
        }
    }
    for index in range(paper_count):
        if include_details:
            questions[f"paper_{index}_type"] = {
                "type": "choice",
                "instructions": f"What kind of research paper is paper {index}?",
                "criteria": PAPER_TYPES,
            }
            questions[f"paper_{index}_difficulty"] = {
                "type": "choice",
                "instructions": f"How difficult would paper {index} be for a developer to implement?",
                "criteria": {
                    "low": "Can be implemented with common libraries and modest background knowledge.",
                    "medium": "Requires specialized ML knowledge, careful evaluation, or moderate engineering.",
                    "high": "Requires substantial research, training, infrastructure, or mathematical background.",
                },
            }
    return questions


def _answer_value(answer: dict | None, key: str, default=None):
    return (answer or {}).get(key, default)


def _apply_answers(papers: list[dict], answers: dict) -> list[dict]:
    enriched = []
    for index, paper in enumerate(papers):
        paper = dict(paper)
        metadata = dict(paper.get("metadata", {}))
        relevance = answers.get("paper_relevance", {})
        probabilities = relevance.get("probabilities", {})
        relevance_probability = probabilities.get(f"paper_{index}")
        if relevance_probability is None:
            # Accept the earlier per-paper score shape for compatibility.
            old_relevance = answers.get(f"paper_{index}_relevance", {})
            old_score = _answer_value(old_relevance, "score")
            relevance_probability = float(old_score) / 4 if old_score is not None else None
            relevance = old_relevance
        paper_type = answers.get(f"paper_{index}_type", {})
        difficulty = answers.get(f"paper_{index}_difficulty", {})
        relevance_score = relevance_probability
        metadata["jev"] = {
            "relevance_score": round(float(relevance_score), 4)
            if relevance_score is not None
            else None,
            "relevance_confidence": _answer_value(relevance, "confidence")
            if relevance.get("choice") == f"paper_{index}"
            else None,
            "paper_type": _answer_value(paper_type, "choice"),
            "paper_type_confidence": _answer_value(paper_type, "confidence"),
            "implementation_difficulty": _answer_value(difficulty, "choice"),
            "difficulty_confidence": _answer_value(difficulty, "confidence"),
        }
        paper["metadata"] = metadata
        enriched.append(paper)
    return enriched


def enrich_papers_with_jev(query: str, papers: list[dict], config: AppConfig | None = None) -> list[dict]:
    """Add optional Jev decisions while preserving the original paper order."""
    config = config or get_config()
    if not papers or not jev_enabled(config):
        return papers

    scored_papers = papers[: config.jev_max_papers]
    cache_key = _cache_key(query, scored_papers, config)
    cached = _cached_result(cache_key)
    if cached is not None:
        return cached + papers[len(scored_papers):]

    payload = {
        "state": _paper_state(query, scored_papers),
        "model": config.jev_model,
        "questions": _questions(len(scored_papers), config.jev_include_details),
    }
    try:
        headers = {"Content-Type": "application/json"}
        if config.jev_api_key:
            headers["Authorization"] = f"Bearer {config.jev_api_key}"
        response = requests.post(
            config.jev_endpoint,
            headers=headers,
            json=payload,
            timeout=config.jev_timeout_seconds,
        )
        response.raise_for_status()
        answers = response.json().get("answers", {})
        enriched = _apply_answers(scored_papers, answers)
        _store_cached_result(cache_key, enriched)
        logger.info("Jev enriched paper_count=%d model=%s", len(scored_papers), config.jev_model)
        return enriched + papers[len(scored_papers):]
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.warning("Jev enrichment unavailable; keeping existing results: %s", exc)
        return papers


def rerank_papers_with_jev(query: str, papers: list[dict], config: AppConfig | None = None) -> list[dict]:
    """Use Jev relevance probabilities to reorder search results when available.

    Papers that Jev did not score stay after scored papers in their original order.
    This keeps the existing hybrid ranking as a safe fallback.
    """
    enriched = enrich_papers_with_jev(query, papers, config)
    scored = [
        (index, paper)
        for index, paper in enumerate(enriched)
        if paper.get("metadata", {}).get("jev", {}).get("relevance_score") is not None
    ]
    if not scored:
        return enriched

    scored_indexes = {index for index, _ in scored}
    ranked_scored = sorted(
        scored,
        key=lambda item: item[1]["metadata"]["jev"]["relevance_score"],
        reverse=True,
    )
    remaining = [paper for index, paper in enumerate(enriched) if index not in scored_indexes]
    result = [paper for _, paper in ranked_scored] + remaining
    logger.info(
        "Jev reranked search scored=%d total=%d top_paper=%s",
        len(ranked_scored),
        len(result),
        result[0].get("paper_id") if result else None,
    )
    return result

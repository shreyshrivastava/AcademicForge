import math
import logging
from collections import Counter
from functools import lru_cache

from backend.retrieval.bm25 import paper_text, tokenize
from backend.retrieval.device import select_retrieval_device
from backend.retrieval.models import RetrievalResult, paper_to_result


MODEL_NAME = "BAAI/bge-small-en-v1.5"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None

    device = select_retrieval_device()
    try:
        logger.info("Loading dense retrieval model %s on %s", MODEL_NAME, device)
        return SentenceTransformer(MODEL_NAME, device=device)
    except Exception as exc:
        if device != "cpu":
            logger.warning("Dense retrieval GPU load failed on %s: %s. Falling back to CPU.", device, exc)
            try:
                return SentenceTransformer(MODEL_NAME, device="cpu")
            except Exception:
                return None
        return None


def _dot(left, right) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _norm(vector) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine(left, right) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return _dot(left, right) / (left_norm * right_norm)


def _lexical_dense_fallback(query: str, papers: list[dict], top_k: int) -> list[RetrievalResult]:
    query_counts = Counter(tokenize(query))
    if not query_counts:
        return []

    scored = []
    for paper in papers:
        paper_counts = Counter(tokenize(paper_text(paper)))
        if not paper_counts:
            continue

        vocabulary = set(query_counts) | set(paper_counts)
        query_vector = [query_counts[token] for token in vocabulary]
        paper_vector = [paper_counts[token] for token in vocabulary]
        score = _cosine(query_vector, paper_vector)
        if score > 0:
            result = paper_to_result(paper)
            result.metadata["dense_backend"] = "lexical_fallback"
            result.metadata["dense_score"] = score
            result.metadata["dense_note"] = (
                "Install sentence-transformers to use BAAI/bge-small-en-v1.5 dense retrieval."
            )
            scored.append((score, result))

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
    results = []
    for rank, (_, result) in enumerate(ranked, start=1):
        result.dense_rank = rank
        results.append(result)
    return results


def dense_search(query: str, papers: list[dict], top_k: int = 50) -> list[RetrievalResult]:
    if not query.strip() or not papers:
        return []

    model = _load_sentence_transformer()
    if model is None:
        return _lexical_dense_fallback(query, papers, top_k)

    texts = [paper_text(paper) for paper in papers]
    try:
        query_embedding = model.encode(query, normalize_embeddings=True)
        paper_embeddings = model.encode(texts, normalize_embeddings=True)
    except Exception:
        return _lexical_dense_fallback(query, papers, top_k)

    scored = []
    for paper, embedding in zip(papers, paper_embeddings):
        score = _dot(query_embedding, embedding)
        result = paper_to_result(paper)
        result.metadata["dense_backend"] = MODEL_NAME
        result.metadata["dense_device"] = str(getattr(model, "device", select_retrieval_device()))
        result.metadata["dense_score"] = score
        scored.append((score, result))

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
    results = []
    for rank, (_, result) in enumerate(ranked, start=1):
        result.dense_rank = rank
        results.append(result)
    return results

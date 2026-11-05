import math
import logging
from collections import Counter
from threading import Lock
import time

from backend.retrieval.bm25 import paper_text, tokenize
from backend.retrieval.device import select_retrieval_device
from backend.retrieval.models import RetrievalResult, paper_to_result


MODEL_NAME = "BAAI/bge-small-en-v1.5"
logger = logging.getLogger(__name__)


def _create_sentence_transformer():
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


_MODEL = None
_RETRY_AT = 0.0
_MODEL_LOCK = Lock()


def _load_sentence_transformer():
    global _MODEL, _RETRY_AT
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        if time.monotonic() < _RETRY_AT:
            return None
        _MODEL = _create_sentence_transformer()
        if _MODEL is None:
            _RETRY_AT = time.monotonic() + 60
        return _MODEL


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
                "Semantic retrieval is unavailable; lexical scoring is being used."
            )
            scored.append((score, result))

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
    results = []
    for rank, (_, result) in enumerate(ranked, start=1):
        result.dense_rank = rank
        results.append(result)
    return results


class DenseIndex:
    """Prepared dense index used by evaluation and reusable fixed-corpus search.

    The application-facing dense_search function keeps its existing behavior.
    This class keeps corpus embeddings resident so benchmark latency excludes
    model loading and document encoding, as a prepared-index measurement should.
    """

    def __init__(self, papers: list[dict]):
        self.papers = papers
        self.model = _load_sentence_transformer()
        self.backend = MODEL_NAME if self.model is not None else "lexical_fallback"
        self._fallback_counts = None
        self._embeddings = None
        if self.model is None:
            self._fallback_counts = [Counter(tokenize(paper_text(paper))) for paper in papers]
        else:
            self._embeddings = self.model.encode(
                [paper_text(paper) for paper in papers],
                normalize_embeddings=True,
            )

    def search(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        if not query.strip() or not self.papers or top_k <= 0:
            return []

        if self.model is None:
            query_counts = Counter(tokenize(query))
            if not query_counts:
                return []
            scored = []
            for paper, paper_counts in zip(self.papers, self._fallback_counts):
                vocabulary = set(query_counts) | set(paper_counts)
                score = _cosine(
                    [query_counts[token] for token in vocabulary],
                    [paper_counts[token] for token in vocabulary],
                )
                if score > 0:
                    result = paper_to_result(paper)
                    result.metadata["dense_backend"] = "lexical_fallback"
                    result.metadata["dense_score"] = score
                    scored.append((score, result))
        else:
            query_embedding = self.model.encode(query, normalize_embeddings=True)
            scored = []
            for paper, embedding in zip(self.papers, self._embeddings):
                score = _dot(query_embedding, embedding)
                result = paper_to_result(paper)
                result.metadata["dense_backend"] = MODEL_NAME
                result.metadata["dense_device"] = str(
                    getattr(self.model, "device", select_retrieval_device())
                )
                result.metadata["dense_score"] = score
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

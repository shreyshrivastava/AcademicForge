import math
import re
from collections import Counter

from backend.retrieval.models import RetrievalResult, paper_to_result


TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+-]*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def paper_text(paper: dict) -> str:
    metadata = paper.get("metadata", {}) or {}
    extra_terms = []
    for key in ("categories", "keywords"):
        value = paper.get(key) or metadata.get(key)
        if isinstance(value, str):
            extra_terms.append(value)
        elif value:
            extra_terms.extend(str(item) for item in value)

    return " ".join(
        [
            paper.get("title", ""),
            paper.get("abstract") or paper.get("summary") or "",
            " ".join(extra_terms),
        ]
    )


def bm25_search(query: str, papers: list[dict], top_k: int = 50) -> list[RetrievalResult]:
    query_tokens = tokenize(query)
    if not query_tokens or not papers:
        return []

    documents = [tokenize(paper_text(paper)) for paper in papers]
    doc_freqs = Counter()
    for tokens in documents:
        doc_freqs.update(set(tokens))

    doc_count = len(documents)
    avg_doc_len = sum(len(tokens) for tokens in documents) / max(doc_count, 1)
    k1 = 1.5
    b = 0.75

    scored = []
    for paper, tokens in zip(papers, documents):
        if not tokens:
            continue

        frequencies = Counter(tokens)
        score = 0.0
        doc_len = len(tokens)
        for token in query_tokens:
            if token not in frequencies:
                continue
            idf = math.log(1 + (doc_count - doc_freqs[token] + 0.5) / (doc_freqs[token] + 0.5))
            numerator = frequencies[token] * (k1 + 1)
            denominator = frequencies[token] + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf * numerator / denominator

        if score > 0:
            result = paper_to_result(paper)
            result.metadata["bm25_score"] = score
            scored.append((score, result))

    ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
    results = []
    for rank, (_, result) in enumerate(ranked, start=1):
        result.bm25_rank = rank
        results.append(result)
    return results

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


class BM25Index:
    """Prepare term statistics once for repeated searches over a fixed corpus.

    Scoring and tie order match bm25_search; labels never enter this index.
    """

    def __init__(self, papers: list[dict]):
        self.papers = papers
        self.frequencies = [Counter(tokenize(paper_text(paper))) for paper in papers]
        self.lengths = [sum(counts.values()) for counts in self.frequencies]
        self.doc_freqs = Counter()
        for counts in self.frequencies:
            self.doc_freqs.update(counts.keys())
        self.avg_doc_len = sum(self.lengths) / max(len(papers), 1)

    def search(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.papers or top_k <= 0:
            return []
        k1, b = 1.5, 0.75
        weights = [(token, math.log(1 + (len(self.papers) - self.doc_freqs[token] + 0.5)
                                   / (self.doc_freqs[token] + 0.5))) for token in query_tokens]
        scored = []
        for paper, frequencies, doc_len in zip(self.papers, self.frequencies, self.lengths):
            if not doc_len:
                continue
            score = 0.0
            for token, idf in weights:
                frequency = frequencies[token]
                if frequency:
                    numerator = frequency * (k1 + 1)
                    denominator = frequency + k1 * (1 - b + b * doc_len / self.avg_doc_len)
                    score += idf * numerator / denominator
            if score > 0:
                scored.append((score, paper))
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        results = []
        for rank, (score, paper) in enumerate(ranked, start=1):
            result = paper_to_result(paper)
            result.metadata["bm25_score"] = score
            result.bm25_rank = rank
            results.append(result)
        return results


def bm25_search(query: str, papers: list[dict], top_k: int = 50) -> list[RetrievalResult]:
    return BM25Index(papers).search(query, top_k)

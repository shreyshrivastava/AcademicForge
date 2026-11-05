import math
import re
from collections import Counter

from backend.retrieval.models import RetrievalResult, paper_to_result


TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+-]*")

# This compact English list is used only by the opt-in benchmark ablation.
# Default application retrieval continues to use tokenize() unchanged.
ENGLISH_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with",
})


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def stem_and_remove_stopwords(text: str) -> list[str]:
    """Apply a small deterministic English suffix stemmer for the ablation only.

    This deliberately avoids a new runtime dependency. It is not a replacement
    for a full linguistic stemmer and is never selected by default in the app.
    """
    tokens = []
    for token in tokenize(text):
        if token in ENGLISH_STOPWORDS:
            continue
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


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

    def __init__(self, papers: list[dict], analyzer=tokenize):
        self.papers = papers
        self.analyzer = analyzer
        self.frequencies = [Counter(analyzer(paper_text(paper))) for paper in papers]
        self.lengths = [sum(counts.values()) for counts in self.frequencies]
        self.doc_freqs = Counter()
        for counts in self.frequencies:
            self.doc_freqs.update(counts.keys())
        self.avg_doc_len = sum(self.lengths) / max(len(papers), 1)

    def search(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        query_tokens = self.analyzer(query)
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

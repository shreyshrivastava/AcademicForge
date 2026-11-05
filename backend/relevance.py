"""Query and paper relevance scoring with field-aware domain checks."""
import logging
import re

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 0.12
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "into", "is", "it", "of", "on", "or", "the", "to", "use", "using",
    "what", "with",
}
DOMAIN_TERMS = {
    "health": {
        "adipose", "bariatric", "bmi", "body", "calorie", "diet", "dietary",
        "exercise", "fat", "health", "healthcare", "lipid", "metabolic",
        "metabolism", "nutrition", "obesity", "overweight", "physical",
        "weight",
    },
    "ai": {
        "ai", "answer", "factuality", "generation", "hallucination",
        "hallucinations", "language", "llm", "llms", "model", "models", "nlp",
        "rag", "retrieval", "transformer",
    },
    "software": {
        "bug", "code", "debugging", "developer", "program", "software",
        "testing", "tests", "verification",
    },
    "energy": {
        "energy", "forecast", "forecasting", "grid", "photovoltaic",
        "renewable", "solar", "wind",
    },
}
STRONG_DOMAIN_TERMS = {
    "health": {
        "adipose", "bariatric", "bmi", "calorie", "cardiac", "clinical",
        "diet", "dietary", "exercise", "healthcare", "lipid", "medical",
        "metabolic", "metabolism", "nutrition", "obesity", "overweight",
        "physical",
    },
    "ai": {"factuality", "hallucination", "hallucinations", "llm", "llms", "nlp", "rag", "retrieval"},
    "software": {"bug", "code", "debugging", "developer", "software", "testing", "tests", "verification"},
    "energy": {"energy", "forecast", "forecasting", "grid", "photovoltaic", "renewable", "solar"},
}
FAT_LOSS_TERMS = {
    "adipose", "bariatric", "bmi", "calorie", "diet", "dietary", "exercise",
    "lipid", "metabolic", "metabolism", "nutrition", "obesity", "overweight",
}
FAT_REDUCTION_TERMS = {
    "bariatric", "calorie", "diet", "dietary", "exercise", "metabolic",
    "metabolism", "nutrition", "obesity", "overweight",
}


def relevance_score(query: str, paper: dict) -> float:
    """Score title + abstract relevance against the user's original query."""
    query_terms = _expanded_query_terms(query)
    if not query_terms:
        return 0.0

    title_terms = set(_content_tokens(paper.get("title", "")))
    abstract_terms = set(_content_tokens(paper.get("abstract", "")))
    paper_terms = title_terms | abstract_terms
    if not paper_terms:
        return 0.0

    overlap = query_terms & paper_terms
    title_overlap = query_terms & title_terms
    score = (len(overlap) / len(query_terms)) + (0.12 * len(title_overlap))

    query_domain = infer_query_domain(query)
    if query_domain != "general":
        domain_terms = DOMAIN_TERMS.get(query_domain, set())
        domain_overlap_count = len(paper_terms & domain_terms)
        strong_overlap_count = len(paper_terms & STRONG_DOMAIN_TERMS.get(query_domain, set()))
        if strong_overlap_count >= 1 or domain_overlap_count >= 3:
            score += 0.2
        elif domain_overlap_count == 1:
            score -= 0.05
        else:
            score -= 0.2

    return round(max(score, 0.0), 4)


def filter_relevant_papers(query: str, papers: list[dict], threshold: float = RELEVANCE_THRESHOLD) -> list[dict]:
    """Discard candidates that are too weakly related to the user's query."""
    query_domain = infer_query_domain(query)
    relevant = []
    for paper in papers:
        paper = dict(paper)
        metadata = dict(paper.get("metadata", {}))
        score = relevance_score(query, paper)
        paper_domain = infer_paper_domain(paper)
        metadata["relevance_score"] = score
        metadata["query_domain"] = query_domain
        metadata["paper_domain"] = paper_domain
        if query_domain != "general" and paper_domain not in {query_domain, "general"}:
            metadata["domain_warning"] = (
                f"Query domain is {query_domain}, but this paper appears to be {paper_domain}."
            )
        paper["metadata"] = metadata
        if (
            score >= threshold
            and not _is_obvious_domain_mismatch(query_domain, paper_domain, score)
            and _has_domain_support(query, query_domain, paper_domain, paper)
        ):
            relevant.append(paper)

    relevant.sort(key=lambda item: item.get("metadata", {}).get("relevance_score", 0), reverse=True)
    logger.info(
        "Relevance gate original_query=%r query_domain=%s threshold=%.2f before=%d after=%d",
        query,
        query_domain,
        threshold,
        len(papers),
        len(relevant),
    )
    if query_domain != "general":
        mismatches = [
            paper for paper in papers
            if _is_obvious_domain_mismatch(
                query_domain,
                paper.get("metadata", {}).get("paper_domain", infer_paper_domain(paper)),
                paper.get("metadata", {}).get("relevance_score", 0),
            )
        ]
        if mismatches and len(mismatches) >= max(2, len(papers) // 3):
            logger.warning(
                "Domain sanity warning query_domain=%s mismatch_count=%d candidate_count=%d",
                query_domain,
                len(mismatches),
                len(papers),
            )
    return relevant


def infer_query_domain(query: str) -> str:
    terms = set(_content_tokens(query))
    scores = {
        domain: len(terms & domain_terms)
        for domain, domain_terms in DOMAIN_TERMS.items()
    }
    domain, score = max(scores.items(), key=lambda item: item[1])
    return domain if score > 0 else "general"


def infer_paper_domain(paper: dict) -> str:
    metadata = paper.get("metadata", {}) or {}
    fields = " ".join(str(item).lower() for item in metadata.get("fields_of_study", []))
    categories = " ".join(str(item).lower() for item in metadata.get("categories", []))
    text_terms = set(_content_tokens(" ".join([paper.get("title", ""), paper.get("abstract", "")])))
    if "cs.se" in categories and len(text_terms & STRONG_DOMAIN_TERMS["health"]) < 2:
        return "software"
    scores = {}
    for domain, domain_terms in DOMAIN_TERMS.items():
        scores[domain] = len(text_terms & domain_terms) + len(text_terms & STRONG_DOMAIN_TERMS.get(domain, set()))
    field_boosted = False
    if any(term in fields for term in ["medicine", "biology", "psychology"]):
        scores["health"] += 2
        field_boosted = True
    if "computer science" in fields or categories.startswith("cs"):
        scores["ai"] += 1
        scores["software"] += 1
        field_boosted = True
    if any(term in fields for term in ["engineering", "environmental science"]):
        scores["energy"] += 1
        field_boosted = True

    domain, score = max(scores.items(), key=lambda item: item[1])
    if score < 2 and not field_boosted:
        return "general"
    return domain if score > 0 else "general"


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_+-]*", (text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _expanded_query_terms(query: str) -> set[str]:
    terms = set(_content_tokens(query))
    domain = infer_query_domain(query)
    if domain != "general":
        terms.update(DOMAIN_TERMS.get(domain, set()))
    if "loss" in terms:
        terms.add("reduce")
        terms.add("reduction")
    if "forecast" in terms:
        terms.add("forecasting")
    if "hallucination" in terms:
        terms.add("hallucinations")
    return terms


def _is_obvious_domain_mismatch(query_domain: str, paper_domain: str, score: float) -> bool:
    if query_domain == "general" or paper_domain == "general":
        return False
    return query_domain != paper_domain and score < 0.45


def _has_domain_support(query: str, query_domain: str, paper_domain: str, paper: dict) -> bool:
    if query_domain == "general":
        return True
    text_terms = set(_content_tokens(" ".join([paper.get("title", ""), paper.get("abstract", "")])))
    query_terms = set(_content_tokens(query))
    if query_domain == "health" and "fat" in query_terms and ({"reduce", "reducing", "reduction", "loss"} & query_terms):
        return bool(text_terms & FAT_REDUCTION_TERMS)
    if query_domain == "health" and "fat" in query_terms:
        return bool(text_terms & FAT_LOSS_TERMS) or (
            "fat" in text_terms and bool(text_terms & STRONG_DOMAIN_TERMS["health"])
        )
    if paper_domain == query_domain:
        return True
    return (
        len(text_terms & STRONG_DOMAIN_TERMS.get(query_domain, set())) >= 1
        or len(text_terms & DOMAIN_TERMS.get(query_domain, set())) >= 3
    )

from urllib.parse import quote_plus, urlparse
from datetime import date
import logging
import re

import arxiv
import requests

from backend.retrieval import hybrid_search, rerank_results
from backend.retrieval.bm25 import bm25_search
from backend.retrieval.dense import dense_search
from backend.retrieval.models import result_to_paper


logger = logging.getLogger(__name__)

INITIAL_CANDIDATE_COUNT = 30
EXPANDED_CANDIDATE_COUNT = 100
RERANK_POOL_SIZE = 30
FINAL_EVIDENCE_TARGET = 10
MIN_RELEVANT_PAPERS = 3
RELEVANCE_THRESHOLD = 0.12

CORE_CATEGORY_QUOTAS = {
    "Foundational": 2,
    "Recent": 2,
    "Implementation Focused": 2,
    "Evaluation Focused": 1,
    "Alternative Approach": 1,
    "Contrarian View": 1,
}

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


def extract_arxiv_id(query: str) -> str | None:
    """Return an arXiv ID from a raw ID or arxiv.org URL."""
    value = query.strip()
    parsed = urlparse(value)
    if parsed.netloc.endswith("arxiv.org"):
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"abs", "pdf"}:
            return path_parts[1].removesuffix(".pdf")

    match = re.fullmatch(r"([a-z-]+/)?\d{7}|\d{4}\.\d{4,5}(v\d+)?", value)
    if match:
        return value.removesuffix(".pdf")

    return None


def search_arxiv_candidates(query: str, max_results: int = INITIAL_CANDIDATE_COUNT) -> tuple[list[dict], bool]:
    """Fetch arXiv candidates. Direct arXiv IDs bypass broad search."""
    arxiv_id = extract_arxiv_id(query)
    if arxiv_id:
        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        logger.info("External search source=arxiv mode=id id=%s", arxiv_id)
    else:
        search = arxiv.Search(query=query, max_results=max_results)
        logger.info(
            "External search source=arxiv query=%r url=%s",
            query,
            "https://export.arxiv.org/api/query?search_query="
            f"{quote_plus(query)}&sortBy=relevance&sortOrder=descending&start=0&max_results={max_results}",
        )

    papers = []
    client = arxiv.Client()
    try:
        results = list(client.results(search))
    except Exception as exc:
        logger.warning(
            "arXiv search failed query=%r direct_id=%s error=%s",
            query,
            bool(arxiv_id),
            exc,
        )
        return [], bool(arxiv_id)

    for result in results:
        try:
            paper_url = result.pdf_url or result.entry_id
            papers.append(
                {
                    "paper_id": result.entry_id,
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "source": "arxiv",
                    "url": paper_url,
                    "link": paper_url,
                    "published": result.published.date().isoformat(),
                    "date": result.published.date().isoformat(),
                    "metadata": {
                        "categories": list(result.categories or []),
                        "primary_category": result.primary_category,
                    },
                }
            )
        except Exception as exc:
            logger.warning("Skipping malformed arXiv result query=%r error=%s", query, exc)

    logger.info("arXiv candidates fetched direct_id=%s count=%d", bool(arxiv_id), len(papers))
    return papers, bool(arxiv_id)


def search_semantic_scholar_candidates(query: str, max_results: int = INITIAL_CANDIDATE_COUNT) -> list[dict]:
    """Fetch broad-domain paper candidates from Semantic Scholar."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": str(max_results),
        "fields": (
            "title,abstract,authors,year,url,externalIds,fieldsOfStudy,"
            "publicationDate,citationCount,influentialCitationCount"
        ),
    }
    logger.info(
        "External search source=semantic_scholar query=%r url=%s?query=%s&limit=%d&fields=%s",
        query,
        url,
        quote_plus(query),
        max_results,
        params["fields"],
    )
    try:
        response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Semantic Scholar search failed query=%r error=%s", query, exc)
        return []

    try:
        results = response.json().get("data", [])
    except ValueError as exc:
        logger.warning("Semantic Scholar returned invalid JSON query=%r error=%s", query, exc)
        return []

    papers = []
    for result in results:
        try:
            title = result.get("title") or ""
            abstract = result.get("abstract") or ""
            if not title or not abstract:
                continue

            paper_url = result.get("url") or ""
            external_ids = result.get("externalIds") or {}
            authors = [
                author.get("name", "")
                for author in result.get("authors", [])
                if author.get("name")
            ]
            papers.append(
                {
                    "paper_id": result.get("paperId") or paper_url or title,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "source": "semantic_scholar",
                    "url": paper_url,
                    "link": paper_url,
                    "published": result.get("publicationDate") or str(result.get("year") or ""),
                    "date": result.get("publicationDate") or str(result.get("year") or ""),
                    "metadata": {
                        "fields_of_study": result.get("fieldsOfStudy") or [],
                        "external_ids": external_ids,
                        "semantic_scholar_year": result.get("year"),
                        "citation_count": result.get("citationCount"),
                        "influential_citation_count": result.get("influentialCitationCount"),
                    },
                }
            )
        except Exception as exc:
            logger.warning("Skipping malformed Semantic Scholar result query=%r error=%s", query, exc)

    logger.info("Semantic Scholar candidates fetched count=%d", len(papers))
    return papers


def search_live_candidates(query: str, max_results: int = INITIAL_CANDIDATE_COUNT) -> tuple[list[dict], bool]:
    """Fetch fresh candidates from external sources for every user query."""
    arxiv_papers, direct_id = search_arxiv_candidates(query, max_results=max_results)
    if direct_id:
        return arxiv_papers, True

    semantic_papers = search_semantic_scholar_candidates(query, max_results=max_results)
    candidates = _dedupe_papers([*semantic_papers, *arxiv_papers])
    logger.info(
        "Live candidates merged original_query=%r total=%d semantic_scholar=%d arxiv=%d",
        query,
        len(candidates),
        len(semantic_papers),
        len(arxiv_papers),
    )
    return candidates, False


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
    """Rank candidate papers with hybrid retrieval and reranking."""
    retrieved = hybrid_search(query, papers, top_k=RERANK_POOL_SIZE)
    ranked = [
        result_to_paper(result)
        for result in rerank_results(query, retrieved, top_k=top_k)
    ]
    logger.info("Papers ranked pool_size=%d", len(ranked))
    return ranked


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


def not_enough_relevant_message() -> str:
    return "Not enough relevant papers found. Try a more specific academic query."


def categorize_paper(paper: dict) -> str:
    """Assign a builder-facing evidence category using transparent heuristics."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    text = f"{title} {abstract}".lower()

    if _contains_any(text, ["survey", "review", "taxonomy", "overview", "systematic literature"]):
        return "Survey"
    if _contains_any(text, ["benchmark", "evaluation", "dataset", "metric", "leaderboard", "testbed"]):
        return "Evaluation Focused"
    if _contains_any(text, ["implementation", "framework", "system", "toolkit", "library", "pipeline", "deploy", "efficient", "lightweight"]):
        return "Implementation Focused"
    if _contains_any(text, ["limitation", "failure", "critique", "pitfall", "challenge", "rethink", "rethinking", "revisiting"]):
        return "Contrarian View"
    if _contains_any(text, ["alternative", "instead", "without", "non-", "contrastive", "different approach"]):
        return "Alternative Approach"
    if _paper_year(paper) >= date.today().year - 1:
        return "Recent"
    return "Foundational"


def select_evidence_set(
    ranked_papers: list[dict],
    target: int = FINAL_EVIDENCE_TARGET,
    preferred_categories: list[str] | None = None,
) -> list[dict]:
    """Select a balanced 8-10 paper evidence set from the reranked pool."""
    selected = []
    selected_ids = set()
    preferred_set = {category for category in (preferred_categories or []) if category}

    annotated = []
    for rank, paper in enumerate(ranked_papers, start=1):
        paper = dict(paper)
        category = categorize_paper(paper)
        paper["metadata"] = dict(paper.get("metadata", {}))
        paper["metadata"]["academicforge_category"] = category
        paper["metadata"]["selection_rank"] = rank
        annotated.append(paper)

    if preferred_set:
        for paper in annotated:
            if paper["metadata"]["academicforge_category"] in preferred_set:
                paper["metadata"]["category_focus_matched"] = True
                _append_unique(selected, selected_ids, paper)
                if len(selected) >= target:
                    return selected

        for paper in annotated:
            paper["metadata"]["category_focus_matched"] = False
            _append_unique(selected, selected_ids, paper)
            if len(selected) >= target:
                break
        return selected

    for category, quota in CORE_CATEGORY_QUOTAS.items():
        for paper in [item for item in annotated if item["metadata"]["academicforge_category"] == category]:
            if len([item for item in selected if item["metadata"]["academicforge_category"] == category]) >= quota:
                break
            _append_unique(selected, selected_ids, paper)
            if len(selected) >= target:
                return selected

    for paper in annotated:
        _append_unique(selected, selected_ids, paper)
        if len(selected) >= target:
            break

    logger.info(
        "Evidence set selected ranked=%d selected=%d categories=%s",
        len(ranked_papers),
        len(selected),
        {
            category: len([paper for paper in selected if paper.get("metadata", {}).get("academicforge_category") == category])
            for category in sorted({paper.get("metadata", {}).get("academicforge_category") for paper in selected})
        },
    )
    return selected


def retrieve_and_rank_papers(query: str, preferred_categories: list[str] | None = None) -> list[dict]:
    """End-to-end retrieval pipeline used by the API."""
    original_query = query.strip()
    search_query = rewrite_search_query(original_query)
    logger.info(
        "Retrieval query original_query=%r rewritten_query=%r source_query=%r",
        original_query,
        search_query,
        search_query,
    )
    candidates, direct_id = search_live_candidates(search_query)
    if direct_id:
        return select_evidence_set(candidates, target=1)
    papers = filter_relevant_papers(original_query, candidates)
    if len(papers) < FINAL_EVIDENCE_TARGET:
        logger.info(
            "Expanding source search original_query=%r relevant_count=%d target=%d expanded_candidates=%d",
            original_query,
            len(papers),
            FINAL_EVIDENCE_TARGET,
            EXPANDED_CANDIDATE_COUNT,
        )
        expanded_candidates, _ = search_live_candidates(
            search_query,
            max_results=EXPANDED_CANDIDATE_COUNT,
        )
        candidates = _dedupe_papers([*candidates, *expanded_candidates])
        papers = filter_relevant_papers(original_query, candidates)
    if len(papers) < MIN_RELEVANT_PAPERS:
        logger.warning(
            "Not enough relevant papers original_query=%r relevant_count=%d",
            original_query,
            len(papers),
        )
        return []
    ranked = rank_papers(query, papers)
    selected = select_evidence_set(ranked, preferred_categories=preferred_categories)
    _log_final_selection(original_query, selected)
    return selected


def debug_retrieval(query: str) -> dict:
    """Return a structured retrieval trace for diagnostics and tests."""
    original_query = query.strip()
    search_query = rewrite_search_query(original_query)
    candidates, direct_id = search_live_candidates(search_query)
    relevant = filter_relevant_papers(original_query, candidates)
    bm25_results = bm25_search(original_query, relevant, top_k=10)
    dense_results = dense_search(original_query, relevant, top_k=10)
    ranked = rank_papers(original_query, relevant) if relevant else []
    selected = select_evidence_set(ranked) if len(relevant) >= MIN_RELEVANT_PAPERS else []
    trace = {
        "original_query": original_query,
        "rewritten_query": search_query,
        "source_query": search_query,
        "direct_id": direct_id,
        "query_domain": infer_query_domain(original_query),
        "external_results": [_paper_debug_row(paper) for paper in candidates[:10]],
        "relevant_results": [_paper_debug_row(paper) for paper in relevant[:10]],
        "bm25_results": [_result_debug_row(result) for result in bm25_results],
        "dense_results": [_result_debug_row(result) for result in dense_results],
        "final_selected": [_paper_debug_row(paper) for paper in selected],
        "message": not_enough_relevant_message() if len(relevant) < MIN_RELEVANT_PAPERS else "",
    }
    logger.info("Retrieval debug trace query=%r trace=%s", original_query, trace)
    return trace


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


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


def _dedupe_papers(papers: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for paper in papers:
        key = (
            paper.get("paper_id")
            or paper.get("url")
            or re.sub(r"\s+", " ", paper.get("title", "").lower()).strip()
        )
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(paper)
    return deduped


def _paper_year(paper: dict) -> int:
    value = str(paper.get("published") or paper.get("date") or "")
    match = re.search(r"\b(19|20)\d{2}\b", value)
    if not match:
        return 0
    return int(match.group(0))


def _append_unique(selected: list[dict], selected_ids: set[str], paper: dict):
    paper_id = paper.get("paper_id") or paper.get("url") or paper.get("link") or paper.get("title")
    if paper_id in selected_ids:
        return
    selected_ids.add(paper_id)
    selected.append(paper)


def _paper_debug_row(paper: dict) -> dict:
    metadata = paper.get("metadata", {}) or {}
    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "score": metadata.get("relevance_score"),
        "source": paper.get("source", ""),
        "source_category": (
            metadata.get("fields_of_study")
            or metadata.get("categories")
            or metadata.get("primary_category")
        ),
        "paper_domain": metadata.get("paper_domain"),
        "url": paper.get("url") or paper.get("link"),
    }


def _result_debug_row(result) -> dict:
    metadata = result.metadata or {}
    return {
        "title": result.title,
        "abstract": result.abstract,
        "score": metadata.get("bm25_score") or metadata.get("dense_score") or result.rrf_score,
        "source": result.source,
        "source_category": (
            metadata.get("fields_of_study")
            or metadata.get("categories")
            or metadata.get("primary_category")
        ),
        "paper_domain": metadata.get("paper_domain"),
        "bm25_rank": result.bm25_rank,
        "dense_rank": result.dense_rank,
        "rrf_score": result.rrf_score,
    }


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

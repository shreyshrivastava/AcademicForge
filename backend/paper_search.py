"""Live candidate search across arXiv and Semantic Scholar."""
from urllib.parse import quote_plus, urlparse
import logging
import os
import re

import xml.etree.ElementTree as ET
import requests

from backend.errors import SourceUnavailable
from backend.sources import diagnostics, get_source, source_status

logger = logging.getLogger(__name__)

INITIAL_CANDIDATE_COUNT = 30


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
    params = {"start": 0, "max_results": min(max_results, 100), "sortBy": "relevance"}
    params["id_list" if arxiv_id else "search_query"] = arxiv_id or query
    try:
        response = get_source("https://export.arxiv.org/api/query", params=params)
        try:
            root = ET.fromstring(response.content)
            if root.tag != "{http://www.w3.org/2005/Atom}feed":
                raise ValueError("Invalid arXiv feed")
        finally:
            response.close()
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            entry_id = entry.findtext("atom:id", "", ns)
            if "/api/errors" in entry_id:
                raise ValueError("arXiv rejected the query")
            title = " ".join(entry.findtext("atom:title", "", ns).split())
            abstract = " ".join(entry.findtext("atom:summary", "", ns).split())
            if not title or not abstract or not entry_id:
                continue
            url = next((link.attrib["href"] for link in entry.findall("atom:link", ns)
                        if link.attrib.get("title") == "pdf" and link.attrib.get("href")), entry_id)
            published = entry.findtext("atom:published", "", ns)[:10]
            primary = entry.find("arxiv:primary_category", ns)
            papers.append({
                "paper_id": entry_id, "title": title, "abstract": abstract,
                "authors": [author.findtext("atom:name", "", ns) for author in entry.findall("atom:author", ns)],
                "source": "arxiv", "url": url, "link": url, "published": published, "date": published,
                "metadata": {"categories": [c.attrib.get("term", "") for c in entry.findall("atom:category", ns)],
                             "primary_category": primary.attrib.get("term", "") if primary is not None else ""},
            })
        source_status("arxiv", "ok")
        return papers, bool(arxiv_id)
    except (requests.RequestException, ET.ParseError, ValueError, SourceUnavailable):
        logger.warning("arXiv source unavailable", exc_info=True)
        source_status("arxiv", "unavailable")
        return [], bool(arxiv_id)


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
    # Anonymous requests share one public rate limit and are usually refused with 429.
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    headers = {"x-api-key": api_key} if api_key else {}
    try:
        response = get_source(url, params=params, headers=headers)
        try:
            payload = response.json()
        finally:
            response.close()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("Invalid Semantic Scholar response")
        results = payload["data"]
        source_status("semantic_scholar", "ok")
    except (requests.RequestException, ValueError, SourceUnavailable) as exc:
        rate_limited = isinstance(exc, requests.HTTPError) and getattr(exc.response, "status_code", None) == 429
        logger.warning("Semantic Scholar source unavailable rate_limited=%s keyed=%s", rate_limited, bool(api_key), exc_info=True)
        source_status("semantic_scholar", "rate_limited" if rate_limited and not api_key else "unavailable")
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
        if diagnostics()["sources"].get("arxiv") == "unavailable":
            raise SourceUnavailable()
        return arxiv_papers, True

    semantic_papers = search_semantic_scholar_candidates(query, max_results=max_results)
    if all(diagnostics()["sources"].get(name) in {"unavailable", "rate_limited"} for name in ("arxiv", "semantic_scholar")):
        raise SourceUnavailable()
    candidates = _dedupe_papers([*semantic_papers, *arxiv_papers])
    logger.info(
        "Live candidates merged original_query=%r total=%d semantic_scholar=%d arxiv=%d",
        query,
        len(candidates),
        len(semantic_papers),
        len(arxiv_papers),
    )
    return candidates, False


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

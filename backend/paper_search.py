"""Live candidate search across arXiv and OpenAlex."""
from urllib.parse import quote_plus, urlparse
import logging
import re

import xml.etree.ElementTree as ET
import requests

from backend.errors import SourceUnavailable
from backend.sources import diagnostics, get_source, source_status

logger = logging.getLogger(__name__)

INITIAL_CANDIDATE_COUNT = 30
OPENALEX_ENABLED = False


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


def _openalex_abstract(inverted_index) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not isinstance(inverted_index, dict):
        return ""
    positions = {pos: word for word, places in inverted_index.items() for pos in places}
    return " ".join(positions[pos] for pos in sorted(positions))


def search_openalex_candidates(query: str, max_results: int = INITIAL_CANDIDATE_COUNT) -> list[dict]:
    """Fetch broad-domain paper candidates from OpenAlex (free, no API key)."""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": str(max_results),
        "select": "id,doi,display_name,publication_year,publication_date,authorships,"
                  "abstract_inverted_index,cited_by_count,primary_location,topics",
    }
    logger.info("External search source=openalex query=%r url=%s?search=%s&per-page=%d", query, url, quote_plus(query), max_results)
    try:
        response = get_source(url, params=params)
        try:
            payload = response.json()
        finally:
            response.close()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Invalid OpenAlex response")
        results = payload["results"]
        source_status("openalex", "ok")
    except (requests.RequestException, ValueError, SourceUnavailable):
        logger.warning("OpenAlex source unavailable", exc_info=True)
        source_status("openalex", "unavailable")
        return []

    papers = []
    for result in results:
        try:
            title = result.get("display_name") or ""
            abstract = _openalex_abstract(result.get("abstract_inverted_index"))
            if not title or not abstract:
                continue
            location = result.get("primary_location") or {}
            paper_url = location.get("landing_page_url") or result.get("doi") or result.get("id") or ""
            authors = [
                (authorship.get("author") or {}).get("display_name", "")
                for authorship in result.get("authorships") or []
                if (authorship.get("author") or {}).get("display_name")
            ]
            fields = []
            for topic in result.get("topics") or []:
                for level in ("field", "subfield"):
                    name = (topic.get(level) or {}).get("display_name")
                    if name and name not in fields:
                        fields.append(name)
            date = result.get("publication_date") or str(result.get("publication_year") or "")
            papers.append(
                {
                    "paper_id": result.get("id") or paper_url or title,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "source": "openalex",
                    "url": paper_url,
                    "link": paper_url,
                    "published": date,
                    "date": date,
                    "metadata": {
                        "fields_of_study": fields,
                        "external_ids": {"DOI": result["doi"]} if result.get("doi") else {},
                        "citation_count": result.get("cited_by_count"),
                    },
                }
            )
        except Exception as exc:
            logger.warning("Skipping malformed OpenAlex result query=%r error=%s", query, exc)

    logger.info("OpenAlex candidates fetched count=%d", len(papers))
    return papers


def search_live_candidates(query: str, max_results: int = INITIAL_CANDIDATE_COUNT) -> tuple[list[dict], bool]:
    """Fetch fresh candidates from external sources for every user query."""
    arxiv_papers, direct_id = search_arxiv_candidates(query, max_results=max_results)
    if direct_id:
        if diagnostics()["sources"].get("arxiv") == "unavailable":
            raise SourceUnavailable()
        return arxiv_papers, True

    # shortcut: OpenAlex is off because its anonymous search is rate-limited; re-enable once an OPENALEX_API_KEY is supported.
    openalex_papers = search_openalex_candidates(query, max_results=max_results) if OPENALEX_ENABLED else []
    sources = ("arxiv", "openalex") if OPENALEX_ENABLED else ("arxiv",)
    if all(diagnostics()["sources"].get(name) == "unavailable" for name in sources):
        raise SourceUnavailable()
    candidates = _dedupe_papers([*openalex_papers, *arxiv_papers])
    logger.info(
        "Live candidates merged original_query=%r total=%d openalex=%d arxiv=%d",
        query,
        len(candidates),
        len(openalex_papers),
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

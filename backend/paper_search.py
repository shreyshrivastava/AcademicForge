# backend/paper_search.py
import arxiv
from dotenv import load_dotenv
from urllib.parse import urlparse
import re

load_dotenv()

def extract_arxiv_id(query):
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

def search_papers(query):
    """Search arXiv for papers related to the research question"""
    try:
        arxiv_id = extract_arxiv_id(query)
        if arxiv_id:
            search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        else:
            search = arxiv.Search(query=query, max_results=10)

        client = arxiv.Client()
        papers = []
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "abstract": result.summary,
                "link": result.pdf_url or result.entry_id,
                "date": result.published.date().isoformat()
            })

        return papers

    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return []

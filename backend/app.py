from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
from urllib.parse import urlparse
import arxiv
import re

from backend.retrieval import hybrid_search, rerank_results
from backend.retrieval.models import result_to_paper
from backend.llm import provider_name, task_model_config
from backend.roadmap_generator import generate_roadmap as generate_ai_roadmap
from backend.roadmap_generator import roadmap_cache_status as get_roadmap_cache_status
from backend.roadmap_generator import stream_roadmap as stream_ai_roadmap
from backend.summarizer import summarize_paper as summarize_paper_with_ai

load_dotenv()

app = FastAPI()

class SearchRequest(BaseModel):
    query: str

class Paper(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    link: str
    date: str
    paper_id: str = ""
    source: str = "arxiv"
    url: str = ""
    published: str = ""
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float = 0.0
    metadata: Dict = Field(default_factory=dict)

class PapersRequest(BaseModel):
    papers: List[Paper]
    summaries: List[str] = []

@app.get("/config")
async def get_config():
    """Return the active local model configuration for display/debugging."""
    return {
        "llm_provider": provider_name(),
        "llm_models": task_model_config(),
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

@app.post("/search")
async def search_papers(request: SearchRequest):
    """Search arXiv for papers related to the research question"""
    try:
        arxiv_id = extract_arxiv_id(request.query)
        if arxiv_id:
            search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        else:
            search = arxiv.Search(query=request.query, max_results=50)

        papers = []
        client = arxiv.Client()
        for result in client.results(search):
            paper_url = result.pdf_url or result.entry_id
            papers.append({
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
            })

        if not arxiv_id:
            retrieved = hybrid_search(request.query, papers, top_k=20)
            papers = [result_to_paper(result) for result in rerank_results(request.query, retrieved, top_k=5)]

        return {"papers": papers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_paper(paper: Paper):
    """Generate a summary of a paper using the configured local model."""
    try:
        return {"summary": summarize_paper_with_ai(paper.model_dump())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roadmap")
async def generate_roadmap(request: PapersRequest):
    """Generate an implementation roadmap based on paper summaries."""
    try:
        papers = [paper.model_dump() for paper in request.papers]
        roadmap = generate_ai_roadmap(papers, request.summaries)
        return {"roadmap": roadmap}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roadmap/cache-status")
async def roadmap_cache_status(request: PapersRequest):
    """Return whether the selected roadmap is already cached."""
    try:
        papers = [paper.model_dump() for paper in request.papers]
        return get_roadmap_cache_status(papers, request.summaries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roadmap/stream")
async def stream_roadmap(request: PapersRequest):
    """Stream an implementation roadmap as it is generated."""
    try:
        papers = [paper.model_dump() for paper in request.papers]
        return StreamingResponse(
            stream_ai_roadmap(papers, request.summaries),
            media_type="text/plain",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

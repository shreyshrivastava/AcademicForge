from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
from urllib.parse import urlparse
import arxiv
import re

from backend.roadmap_generator import generate_roadmap as generate_ai_roadmap
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

class PapersRequest(BaseModel):
    papers: List[Paper]
    summaries: List[str] = []

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
            search = arxiv.Search(query=request.query, max_results=10)

        papers = []
        client = arxiv.Client()
        for result in client.results(search):
            papers.append(Paper(
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                link=result.pdf_url or result.entry_id,
                date=result.published.date().isoformat()
            ))
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

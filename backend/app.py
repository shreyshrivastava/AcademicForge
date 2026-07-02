from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
import logging

from backend.config import get_config as get_app_config
from backend.data_pipeline import extract_arxiv_id, retrieve_and_rank_papers
from backend.llm import provider_name, task_model_config
from backend.roadmap_generator import generate_roadmap as generate_ai_roadmap
from backend.roadmap_generator import generate_paper_roadmap as generate_ai_paper_roadmap
from backend.roadmap_generator import paper_roadmap_cache_status as get_paper_roadmap_cache_status
from backend.roadmap_generator import roadmap_cache_status as get_roadmap_cache_status
from backend.roadmap_generator import stream_roadmap as stream_ai_roadmap
from backend.summarizer import summarize_paper as summarize_paper_with_ai

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("backend").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI()
logger = logging.getLogger(__name__)

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
    payload = get_app_config().as_public_dict()
    payload["llm_provider"] = provider_name()
    payload["llm_models"] = task_model_config()
    return payload

@app.post("/search")
async def search_papers(request: SearchRequest):
    """Search arXiv for papers related to the research question"""
    try:
        logger.info("Search requested query_preview=%r", request.query[:80])
        arxiv_id = extract_arxiv_id(request.query)
        papers = retrieve_and_rank_papers(request.query)
        logger.info("Search completed arxiv_id=%s result_count=%d", bool(arxiv_id), len(papers))
        return {"papers": papers}
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_paper(paper: Paper):
    """Generate a summary of a paper using the configured local model."""
    try:
        logger.info("Summary requested paper_id=%r title=%r", paper.paper_id, paper.title[:80])
        return {"summary": summarize_paper_with_ai(paper.model_dump())}
    except Exception as e:
        logger.exception("Summary failed paper_id=%r", paper.paper_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roadmap")
async def generate_roadmap(request: PapersRequest):
    """Generate an implementation roadmap based on paper summaries."""
    try:
        logger.info("Mixed roadmap requested paper_count=%d", len(request.papers))
        papers = [paper.model_dump() for paper in request.papers]
        roadmap = generate_ai_roadmap(papers, request.summaries)
        return {"roadmap": roadmap}
    except Exception as e:
        logger.exception("Mixed roadmap failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roadmap/paper")
async def generate_paper_roadmap(paper: Paper):
    """Generate a practical roadmap for one paper."""
    try:
        logger.info("Paper roadmap requested paper_id=%r title=%r", paper.paper_id, paper.title[:80])
        return {"roadmap": generate_ai_paper_roadmap(paper.model_dump())}
    except Exception as e:
        logger.exception("Paper roadmap failed paper_id=%r", paper.paper_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roadmap/paper/cache-status")
async def paper_roadmap_cache_status(paper: Paper):
    """Return whether a single-paper roadmap is already cached."""
    try:
        status = get_paper_roadmap_cache_status(paper.model_dump())
        logger.info(
            "Paper roadmap cache status paper_id=%r cache=%s cached=%s",
            paper.paper_id,
            status.get("cache"),
            status.get("cached"),
        )
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roadmap/cache-status")
async def roadmap_cache_status(request: PapersRequest):
    """Return whether the selected roadmap is already cached."""
    try:
        papers = [paper.model_dump() for paper in request.papers]
        status = get_roadmap_cache_status(papers, request.summaries)
        logger.info(
            "Mixed roadmap cache status paper_count=%d cache=%s cached=%s",
            len(request.papers),
            status.get("cache"),
            status.get("cached"),
        )
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roadmap/stream")
async def stream_roadmap(request: PapersRequest):
    """Stream an implementation roadmap as it is generated."""
    try:
        logger.info("Mixed roadmap stream requested paper_count=%d", len(request.papers))
        papers = [paper.model_dump() for paper in request.papers]
        return StreamingResponse(
            stream_ai_roadmap(papers, request.summaries),
            media_type="text/plain",
        )
    except Exception as e:
        logger.exception("Mixed roadmap stream failed")
        raise HTTPException(status_code=500, detail=str(e))

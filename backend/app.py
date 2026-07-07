from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
import logging
import os

from backend.config import get_config as get_app_config
from backend.data_pipeline import extract_arxiv_id, not_enough_relevant_message, retrieve_and_rank_papers
from backend.llm import provider_name, task_model_config
from backend.research_plan_generator import generate_paper_guidance as generate_ai_paper_guidance
from backend.research_plan_generator import generate_research_plan as generate_ai_research_plan
from backend.research_plan_generator import paper_guidance_cache_status as get_paper_guidance_cache_status
from backend.research_plan_generator import research_plan_cache_status as get_research_plan_cache_status
from backend.research_plan_generator import stream_research_plan as stream_ai_research_plan
from backend.summarizer import summarize_paper as summarize_paper_with_ai

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("backend").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI()
logger = logging.getLogger(__name__)
FAST_MODE_MODEL = os.getenv("ACADEMICFORGE_FAST_MODEL", "mlx-community/Qwen3-4B-4bit")
DEEP_MODE_MODEL = os.getenv("ACADEMICFORGE_DEEP_MODEL", "mlx-community/gemma-4-e2b-it-OptiQ-4bit")

class SearchRequest(BaseModel):
    query: str
    categories: List[str] = []

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
    generation_mode: str = "fast"

class PapersRequest(BaseModel):
    papers: List[Paper]
    summaries: List[str] = []
    query: str = ""
    generation_mode: str = "fast"

def normalize_research_categories(categories: List[str] | None) -> List[str]:
    """Keep Balanced exclusive for API callers too."""
    cleaned = [category for category in (categories or []) if category]
    if "Balanced" in cleaned:
        return []
    return cleaned

def model_for_mode(mode: str | None) -> str:
    return DEEP_MODE_MODEL if (mode or "").strip().lower() == "deep" else FAST_MODE_MODEL

@app.get("/config")
async def get_config():
    """Return the active local model configuration for display/debugging."""
    payload = get_app_config().as_public_dict()
    payload["llm_provider"] = provider_name()
    payload["llm_models"] = task_model_config()
    payload["generation_modes"] = {
        "fast": {
            "label": "Fast Mode (Qwen)",
            "model": FAST_MODE_MODEL,
            "purpose": "Quick insights and shorter responses.",
        },
        "deep": {
            "label": "Deep Mode (Gemma)",
            "model": DEEP_MODE_MODEL,
            "purpose": "Detailed analysis and prototype guidance.",
        },
    }
    return payload

@app.post("/search")
async def search_papers(request: SearchRequest):
    """Search arXiv for papers related to the research question"""
    try:
        logger.info("Search requested query_preview=%r", request.query[:80])
        arxiv_id = extract_arxiv_id(request.query)
        papers = retrieve_and_rank_papers(request.query, normalize_research_categories(request.categories))
        logger.info("Search completed arxiv_id=%s result_count=%d", bool(arxiv_id), len(papers))
        return {
            "papers": papers,
            "message": not_enough_relevant_message() if not papers else "",
        }
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_paper(paper: Paper):
    """Generate a summary of a paper using the configured local model."""
    try:
        logger.info("Summary requested paper_id=%r title=%r", paper.paper_id, paper.title[:80])
        return {"summary": summarize_paper_with_ai(paper.model_dump(), model=FAST_MODE_MODEL)}
    except Exception as e:
        logger.exception("Summary failed paper_id=%r", paper.paper_id)
        raise HTTPException(status_code=500, detail=str(e))

def research_plan_response(text: str) -> dict:
    return {"research_plan": text, "roadmap": text}


def paper_guidance_response(text: str) -> dict:
    return {"guidance": text, "roadmap": text}


@app.post("/research-plan")
async def generate_research_plan(request: PapersRequest):
    """Generate a Research Plan based on selected papers and summaries."""
    try:
        logger.info("Research Plan requested paper_count=%d", len(request.papers))
        papers = [paper.model_dump() for paper in request.papers]
        research_plan = generate_ai_research_plan(
            papers,
            request.summaries,
            request.query,
            model=model_for_mode(request.generation_mode),
        )
        return research_plan_response(research_plan)
    except Exception as e:
        logger.exception("Research Plan failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/roadmap")
async def generate_roadmap(request: PapersRequest):
    return await generate_research_plan(request)


@app.post("/paper-guidance")
async def generate_paper_guidance(paper: Paper):
    """Generate practical guidance for one paper."""
    try:
        logger.info("Paper guidance requested paper_id=%r title=%r", paper.paper_id, paper.title[:80])
        guidance = generate_ai_paper_guidance(paper.model_dump(), model=model_for_mode(paper.generation_mode))
        return paper_guidance_response(guidance)
    except Exception as e:
        logger.exception("Paper guidance failed paper_id=%r", paper.paper_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/roadmap/paper")
async def generate_paper_roadmap(paper: Paper):
    return await generate_paper_guidance(paper)


@app.post("/paper-guidance/cache-status")
async def paper_guidance_cache_status(paper: Paper):
    """Return whether single-paper guidance is already cached."""
    try:
        status = get_paper_guidance_cache_status(paper.model_dump(), model=model_for_mode(paper.generation_mode))
        logger.info(
            "Paper guidance cache status paper_id=%r cache=%s cached=%s",
            paper.paper_id,
            status.get("cache"),
            status.get("cached"),
        )
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/roadmap/paper/cache-status")
async def paper_roadmap_cache_status(paper: Paper):
    return await paper_guidance_cache_status(paper)


@app.post("/research-plan/cache-status")
async def research_plan_cache_status(request: PapersRequest):
    """Return whether the selected Research Plan is already cached."""
    try:
        papers = [paper.model_dump() for paper in request.papers]
        status = get_research_plan_cache_status(
            papers,
            request.summaries,
            request.query,
            model=model_for_mode(request.generation_mode),
        )
        logger.info(
            "Research Plan cache status paper_count=%d cache=%s cached=%s",
            len(request.papers),
            status.get("cache"),
            status.get("cached"),
        )
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/roadmap/cache-status")
async def roadmap_cache_status(request: PapersRequest):
    return await research_plan_cache_status(request)


@app.post("/research-plan/stream")
async def stream_research_plan(request: PapersRequest):
    """Stream a Research Plan as it is generated."""
    try:
        logger.info("Research Plan stream requested paper_count=%d", len(request.papers))
        papers = [paper.model_dump() for paper in request.papers]
        return StreamingResponse(
            stream_ai_research_plan(
                papers,
                request.summaries,
                request.query,
                model=model_for_mode(request.generation_mode),
            ),
            media_type="text/plain",
        )
    except Exception as e:
        logger.exception("Research Plan stream failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/roadmap/stream")
async def stream_roadmap(request: PapersRequest):
    return await stream_research_plan(request)

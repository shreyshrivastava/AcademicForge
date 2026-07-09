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
from backend.research_plan_generator import generate_research_plan as generate_ai_research_plan
from backend.research_plan_generator import research_plan_cache_status as get_research_plan_cache_status
from backend.research_plan_generator import stream_research_plan as stream_ai_research_plan
from backend.guidance_generator import generate_paper_guidance as generate_ai_paper_guidance
from backend.guidance_generator import paper_guidance_cache_status as get_paper_guidance_cache_status
from backend.summarizer import summarize_paper as summarize_paper_with_ai

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("backend").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI()
logger = logging.getLogger(__name__)
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

def model_for_summary(mode: str | None) -> str:
    return get_app_config().model_for_task_and_mode("summary", mode)


def model_for_guidance(mode: str | None) -> str:
    return get_app_config().model_for_task_and_mode("guidance", mode)


def model_for_research_plan(mode: str | None) -> str:
    return get_app_config().model_for_task_and_mode("research_plan", mode)

@app.get("/health")
async def health_check():
    """Health check endpoint returning model and provider details."""
    config = get_app_config()
    return {
        "status": "ok",
        "backend": os.getenv("BACKEND_MODE", "unknown"),
        "provider": provider_name(),
        "model": config.llm_model
    }

@app.get("/config")
async def get_config():
    """Return the active local model configuration for display/debugging."""
    config = get_app_config()
    payload = config.as_public_dict()
    payload["llm_provider"] = provider_name()
    payload["llm_models"] = task_model_config()
    
    fast_model_name = config.llm_model.split("/")[-1]
    deep_model_name = config.llm_research_plan_model.split("/")[-1]
    
    payload["generation_modes"] = {
        "fast": {
            "label": f"Fast Mode ({fast_model_name})",
            "model": config.llm_model,
            "purpose": "Quick insights and shorter responses.",
        },
        "deep": {
            "label": f"Deep Mode ({deep_model_name})",
            "model": config.llm_research_plan_model,
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
        model = model_for_summary(paper.generation_mode)
        return {"summary": summarize_paper_with_ai(paper.model_dump(), model=model, mode=paper.generation_mode)}
    except Exception as e:
        logger.exception("Summary failed paper_id=%r", paper.paper_id)
        raise HTTPException(status_code=500, detail=str(e))

def research_plan_response(text: str) -> dict:
    return {"research_plan": text}


def paper_guidance_response(text: str) -> dict:
    return {"guidance": text}


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
            model=model_for_research_plan(request.generation_mode),
            mode=request.generation_mode,
        )
        return research_plan_response(research_plan)
    except Exception as e:
        logger.exception("Research Plan failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/paper-guidance")
async def generate_paper_guidance(paper: Paper):
    """Generate practical guidance for one paper."""
    try:
        logger.info("Paper guidance requested paper_id=%r title=%r", paper.paper_id, paper.title[:80])
        guidance = generate_ai_paper_guidance(
            paper.model_dump(),
            model=model_for_guidance(paper.generation_mode),
            mode=paper.generation_mode,
        )
        return paper_guidance_response(guidance)
    except Exception as e:
        logger.exception("Paper guidance failed paper_id=%r", paper.paper_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/paper-guidance/cache-status")
async def paper_guidance_cache_status(paper: Paper):
    """Return whether single-paper guidance is already cached."""
    try:
        status = get_paper_guidance_cache_status(
            paper.model_dump(),
            model=model_for_guidance(paper.generation_mode),
            mode=paper.generation_mode,
        )
        logger.info(
            "Paper guidance cache status paper_id=%r cache=%s cached=%s",
            paper.paper_id,
            status.get("cache"),
            status.get("cached"),
        )
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research-plan/cache-status")
async def research_plan_cache_status(request: PapersRequest):
    """Return whether the selected Research Plan is already cached."""
    try:
        papers = [paper.model_dump() for paper in request.papers]
        status = get_research_plan_cache_status(
            papers,
            request.summaries,
            request.query,
            model=model_for_research_plan(request.generation_mode),
            mode=request.generation_mode,
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
                model=model_for_research_plan(request.generation_mode),
                mode=request.generation_mode,
            ),
            media_type="text/plain",
        )
    except Exception as e:
        logger.exception("Research Plan stream failed")
        raise HTTPException(status_code=500, detail=str(e))

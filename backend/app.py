"""AcademicForge API with bounded work and explicit failure contracts."""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import secrets
import time
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

load_dotenv(os.getenv("ACADEMICFORGE_ENV_FILE", ".env"))

from backend.config import get_config as get_app_config
from backend.data_pipeline import not_enough_relevant_message, retrieve_and_rank_papers
from backend.errors import ServiceError
from backend.execution import GENERATION_CAPACITY, SEARCH_CAPACITY, execute
from backend.health import readiness, warmup
from backend.http import RequestBoundary
from backend.llm import provider_name, task_model_config, generation_trace
from backend.runtime import runtime_version_payload
from backend.research_plan_generator import generate_research_plan as generate_ai_research_plan
from backend.research_plan_generator import stream_research_plan as stream_ai_research_plan
from backend.guidance_generator import generate_paper_guidance as generate_ai_paper_guidance
from backend.summarizer import summarize_paper as summarize_paper_with_ai
from backend.schemas import (
    SearchRequest, Paper, PapersRequest, SearchResponse, SummaryResponse,
    GuidanceResponse, ResearchPlanResponse,
)
from backend.streaming import events
from backend.sources import diagnostics

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def startup_event():
    get_app_config()  # Fail early on invalid settings.
    await run_in_threadpool(warmup)


@asynccontextmanager
async def lifespan(app):
    get_app_config()
    task = asyncio.create_task(startup_event())
    try:
        yield
    finally:
        # Model loading cannot safely be killed mid-call. Wait for the worker to finish.
        await task


app = FastAPI(title="AcademicForge", version="0.2.0", lifespan=lifespan)
app.add_middleware(RequestBoundary)


def error_response(request, status, code, message):
    return JSONResponse(status_code=status, content={
        "detail": message,
        "error": {"code": code, "request_id": getattr(request.state, "request_id", "")},
    }, headers={"Retry-After": "2"} if status in {429, 503} else {})


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = uuid4().hex
    started = time.perf_counter()
    key = os.getenv("ACADEMICFORGE_API_KEY", "")
    if key and request.url.path not in {"/health", "/live"}:
        if not secrets.compare_digest(request.headers.get("X-API-Key", "").encode(), key.encode()):
            response = error_response(request, 401, "unauthorized", "A valid API key is required.")
            response.headers["X-Request-ID"] = request.state.request_id
            return response
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request failure request_id=%s", request.state.request_id)
        response = error_response(request, 500, "internal_error", "The request could not be completed.")
    response.headers["X-Request-ID"] = request.state.request_id
    logger.info("request_id=%s method=%s path=%s status=%d elapsed_ms=%.1f",
                request.state.request_id, request.method, request.url.path, response.status_code,
                (time.perf_counter() - started) * 1000)
    return response


@app.exception_handler(ServiceError)
async def service_error(request, exc):
    return error_response(request, exc.status_code, exc.code, exc.public_message)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Avoid echoing paper text or other submitted content in validation errors.
    return error_response(request, 422, "invalid_request", "Invalid input. Check field values and size limits.")


def normalize_research_categories(categories):
    cleaned = [category for category in (categories or []) if category]
    return [] if "Balanced" in cleaned else cleaned


def model_for_summary(mode):
    return get_app_config().model_for_task_and_mode("summary", mode)


def model_for_guidance(mode):
    return get_app_config().model_for_task_and_mode("guidance", mode)


def model_for_research_plan(mode):
    return get_app_config().model_for_task_and_mode("research_plan", mode)


@app.get("/live")
async def live():
    return {"status": "ok"}


@app.get("/health")
async def health_check():
    config = get_app_config()
    payload = {**readiness(), "provider": config.llm_provider, "model": config.llm_model,
               "instance_id": os.getenv("ACADEMICFORGE_INSTANCE_ID", "direct") }
    return JSONResponse(payload, status_code=200 if payload["models_ready"] else 503)


@app.get("/version")
async def version_check():
    return await run_in_threadpool(runtime_version_payload, get_app_config())


@app.get("/config")
async def get_config():
    config = get_app_config()
    payload = config.as_public_dict()
    payload["llm_provider"] = provider_name()
    payload["llm_models"] = task_model_config()
    payload["generation_modes"] = {
        mode: {"label": f"{mode.title()} Mode ({model.split('/')[-1]})", "model": model}
        for mode, model in (("fast", config.llm_model), ("deep", config.llm_research_plan_model))
    }
    return payload


def _search(request):
    papers = retrieve_and_rank_papers(request.query, normalize_research_categories(request.categories))
    state = diagnostics()
    if any(paper.get("metadata", {}).get("dense_backend") == "lexical_fallback" for paper in papers):
        state["warnings"].append("Semantic ranking is unavailable; keyword scoring is being used.")
    if any(paper.get("metadata", {}).get("reranker_backend") == "original_order" for paper in papers):
        state["warnings"].append("The reranker is unavailable; the previous ranking is being used.")
    return {"papers": papers, "message": not_enough_relevant_message() if not papers else "", **state}


@app.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    return await execute(SEARCH_CAPACITY, _search, request)


def generated(field, operation, *args, **kwargs):
    trace = {}
    generation_trace.set(trace)
    return {field: operation(*args, **kwargs), "generation": trace}


@app.post("/summarize", response_model=SummaryResponse)
async def summarize_paper(paper: Paper):
    return await execute(GENERATION_CAPACITY, generated, "summary", summarize_paper_with_ai, paper.model_dump(),
                         model=model_for_summary(paper.generation_mode), mode=paper.generation_mode)


def research_plan_response(text):
    return {"research_plan": text}


def paper_guidance_response(text):
    return {"guidance": text}


@app.post("/research-plan", response_model=ResearchPlanResponse)
async def generate_research_plan(request: PapersRequest):
    return await execute(GENERATION_CAPACITY, generated, "research_plan", generate_ai_research_plan,
                         [paper.model_dump() for paper in request.papers], request.summaries, request.query,
                         model=model_for_research_plan(request.generation_mode), mode=request.generation_mode)


@app.post("/paper-guidance", response_model=GuidanceResponse)
async def generate_paper_guidance(paper: Paper):
    return await execute(GENERATION_CAPACITY, generated, "guidance", generate_ai_paper_guidance, paper.model_dump(),
                         model=model_for_guidance(paper.generation_mode), mode=paper.generation_mode)


@app.post("/research-plan/stream")
async def stream_research_plan(request: PapersRequest, http_request: Request):
    lease = GENERATION_CAPACITY.reserve()
    try:
        iterator = iter(stream_ai_research_plan(
            [paper.model_dump() for paper in request.papers], request.summaries, request.query,
            model=model_for_research_plan(request.generation_mode), mode=request.generation_mode,
        ))
        return StreamingResponse(events(iterator, lease, http_request.state.request_id),
                                 media_type="application/x-ndjson", headers={"Cache-Control": "no-store"},
                                 )
    except BaseException:
        lease.release()
        raise

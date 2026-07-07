# Architecture

AcademicForge is intentionally small: a FastAPI backend, a Streamlit frontend, in-memory retrieval, local LLM generation, and local disk caching.

## High-Level Flow

```text
Streamlit UI
  -> FastAPI backend
  -> arXiv search
  -> retrieval layer
       -> BM25 keyword search
       -> dense semantic search
       -> reciprocal rank fusion
       -> reranker placeholder
  -> selected papers
  -> local MLX summarization
  -> compact Research Plan context
  -> streamed local MLX Research Plan
  -> local JSON cache
```

## Backend

`backend/app.py` exposes the main API:

- `GET /config` - active model configurations
- `POST /search` - arXiv collection plus hybrid retrieval
- `POST /summarize` - paper summary generation
- `POST /research-plan` - non-streaming Research Plan generation
- `POST /research-plan/stream` - streaming Research Plan generation
- `POST /research-plan/cache-status` - cache status for selected Research Plan inputs
- `POST /paper-guidance` - single-paper practical guidance

The older `/roadmap/*` routes remain as compatibility aliases during the product rename.

## Retrieval Layer

The retrieval layer lives in `backend/retrieval/`.

- `bm25.py` searches exact terms across title, abstract, categories, and keywords.
- `dense.py` uses `BAAI/bge-small-en-v1.5` through `sentence-transformers` when available.
- `rrf.py` combines ranked lists with reciprocal rank fusion.
- `hybrid.py` runs BM25, dense search, and RRF together.
- `reranker.py` is a placeholder for future cross-encoder reranking.

RRF is used because BM25 and dense search produce different score distributions. RRF combines rankings without requiring score normalization.

## Local LLM Layer

`backend/llm.py` wraps local generation.

Supported provider:

- `mlx`

Current default model (Fast Mode):
- `mlx-community/Qwen3-4B-4bit`

Task-specific model routing:

- `LOCAL_LLM_MODEL`
- `LOCAL_LLM_SUMMARY_MODEL`
- `LOCAL_LLM_RESEARCH_PLAN_MODEL`

This lets the project use communicator/instruct models for summaries and Research Plans while reserving coder models for future code-generation workflows.

Provider notes:

- `mlx` is intended for Apple Silicon and uses `mlx-lm`.
- CUDA and ROCm are future migration targets, but they are not active runtime providers yet.

## Caching

`backend/cache.py` stores JSON cache entries under `.academicforge_cache/` by default.

Cached items:

- paper summaries
- generated Research Plans

The cache key includes the task, model, selected papers, summaries, and compact Research Plan context, so changing model or selected papers produces a new cache entry.

## Frontend

`frontend/streamlit_app.py` provides a simple judge-friendly UI:

- search box
- Research Lens control
- ranked paper cards with category chips
- inline Summary and Guidance panels
- summary generation
- streamed Research Plan rendering
- Markdown export

## Why This Architecture

The project avoids databases and hosted services for the hackathon MVP. Everything runs locally, is easy to explain, and can be tested with small scripts. The architecture keeps feature boundaries clear while leaving room for future sources, rerankers, and deployment paths.

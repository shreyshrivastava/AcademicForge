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
  -> compact roadmap context
  -> streamed local MLX roadmap
  -> local JSON cache
```

## Backend

`backend/app.py` exposes the main API:

- `GET /config` - active local model configuration
- `POST /search` - arXiv collection plus hybrid retrieval
- `POST /summarize` - local paper summary generation
- `POST /roadmap` - non-streaming roadmap generation
- `POST /roadmap/stream` - streaming roadmap generation
- `POST /roadmap/cache-status` - cache status for selected roadmap inputs

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

Current provider:

- `mlx`

Current default model:

- `mlx-community/Qwen3-4B-4bit`

Task-specific model routing:

- `LOCAL_LLM_MODEL`
- `LOCAL_LLM_SUMMARY_MODEL`
- `LOCAL_LLM_ROADMAP_MODEL`

This lets the project use communicator/instruct models for summaries and roadmaps while reserving coder models for future code-generation workflows.

## Caching

`backend/cache.py` stores JSON cache entries under `.academicforge_cache/` by default.

Cached items:

- paper summaries
- generated roadmaps

The cache key includes the task, model, selected papers, summaries, and compact roadmap context, so changing model or selected papers produces a new cache entry.

## Frontend

`frontend/streamlit_app.py` provides a simple judge-friendly UI:

- search box
- ranked paper table
- paper detail expanders
- paper selection control
- summary generation
- streamed roadmap rendering
- Markdown export

## Why This Architecture

The project avoids databases and hosted services for the hackathon MVP. Everything runs locally, is easy to explain, and can be tested with small scripts. The architecture keeps feature boundaries clear while leaving room for future sources, rerankers, and deployment paths.

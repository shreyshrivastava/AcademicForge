# Architecture

AcademicForge is two processes: a FastAPI backend (`backend/app.py`) and a Streamlit interface (`frontend/streamlit_app.py`). The interface reaches the backend at `ACADEMICFORGE_BACKEND_URL`. The standalone demo (`frontend/streamlit_cloud_app.py`) runs the same interface with an in-process adapter and no backend.

## Request flow

```text
Streamlit interface
  -> POST /search
       paper_search.py   live arXiv search; Semantic Scholar too when SEMANTIC_SCHOLAR_API_KEY is set
       relevance.py      drop off-topic papers, check the chosen field
       data_pipeline.py  rank: score fusion of stemmed BM25 + BGE dense (alpha 0.9)
                         optional dense-only, RRF, cross-encoder, OpenJev
       evidence.py       pick a balanced set of up to 10 papers
  -> POST /summarize, /paper-guidance     one paper at a time
  -> POST /research-plan/stream           the selected papers and their summaries
       research_plan_generator.py         prompt, citations, "Selected evidence" key
       llm.py                             route to the local or cloud model
```

## Backend modules

| Module | Job |
| :--- | :--- |
| `app.py` | Routes, request limits, API key check, error codes. |
| `data_pipeline.py` | Orchestrates a search: rewrite the query, fetch, filter, rank, select, cache. |
| `paper_search.py` | arXiv and Semantic Scholar clients and de-duplication. |
| `relevance.py` | Relevance scores and domain checks. |
| `evidence.py` | Sorts papers into categories (Foundational, Survey and so on) and picks the final set. |
| `search_context.py` | The Field presets and how they rewrite a query. |
| `sources.py` | Per-source status, warnings, and the shared retry and timeout budget. |
| `retrieval/` | `bm25.py`, `dense.py` (BGE), `rrf.py`, `reranker.py` (BGE cross-encoder), `device.py`. |
| `jev.py` | Optional OpenJev rescoring of the top papers in Quality search mode. |
| `llm.py` | Picks a backend for each request and records which one answered. |
| `llm_local.py`, `llm_remote.py` | MLX and Transformers backends; Groq-style and Fireworks clients. |
| `llm_text.py`, `llm_common.py` | Output cleanup, the shared plain-writing rule for prompts, shared settings. |
| `summarizer.py`, `guidance_generator.py`, `research_plan_generator.py` | The three prompts. |
| `health.py`, `runtime.py` | Warmup and readiness per component. |

## Ranking

Score fusion of stemmed BM25 and `BAAI/bge-small-en-v1.5` (dense weight 0.9) is the default. `ACADEMICFORGE_RETRIEVAL_MODE` switches to `dense`, `hybrid` (RRF) or `hybrid_rerank` (RRF plus the `BAAI/bge-reranker-base` cross-encoder). The choice and its evidence are in [the benchmark guide](benchmarks/README.md).

OpenJev runs only in Quality search mode with `JEV_ENABLED=true`. It asks a local or hosted decision model about relevance, paper type and difficulty for the top few papers, caches the answer, and keeps the original order if the call fails or returns something malformed. It is not part of the measured default.

## Models

`LOCAL_LLM_PROVIDER=auto` picks MLX (`mlx-community/gemma-2-2b-it-4bit`) on Apple Silicon and Transformers (`google/gemma-2-2b-it`) elsewhere. Transformers covers both NVIDIA CUDA and AMD ROCm through one code path, because PyTorch exposes AMD GPUs through its CUDA device API. That's also why `/version` reports `retrieval_device: cuda` on ROCm.

Deep mode can use Groq (`REMOTE_LLM_USE_FOR_DEEP=true`) or Fireworks DeepSeek (`LOCAL_LLM_USE_FIREWORKS=true`). If a cloud call fails before any output, the request falls back to the local model and the response says so.

## Interface

The interface is one column: the question box with Field, Lens and Model, then the ranked paper cards (Use in plan, Summary, Guidance), then the research plan. The papers and plan section is a Streamlit fragment, so ticking a paper or asking for a summary reruns only that section. The backend config is reused for 30 seconds per session. Colors, type and copy rules are in [DESIGN.md](../DESIGN.md).

## Reliability

- Request schemas limit input size before any model or network work.
- Search and generation run in worker threads behind separate admission limits.
- Each source reports its own status, so a partial outage shows a warning and a total one returns HTTP 503.
- Streams are NDJSON and always end in `done` or `error`. A broken stream is never silently replayed.
- Public errors carry a stable code and a request ID.

See [reliability](reliability.md) for the full list and its limits.

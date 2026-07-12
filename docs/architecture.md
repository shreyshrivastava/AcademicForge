# Architecture

AcademicForge is an AMD notebook-oriented research-to-plan system. It uses a
Streamlit frontend, a FastAPI backend, live academic retrieval, hybrid ranking,
and Transformers-based local inference for ROCm-compatible AMD GPU deployment.

## High-Level Flow

```text
Streamlit UI
  -> FastAPI backend
  -> live source search
       -> arXiv
       -> Semantic Scholar
  -> candidate paper pool
  -> retrieval layer
       -> BM25 lexical ranking
       -> dense semantic ranking
       -> reciprocal rank fusion
       -> cross-encoder reranking
  -> Research Lens selection
  -> ranked paper cards
       -> on-demand Summary
       -> on-demand Guidance
       -> selected papers
  -> compact Research Plan context
  -> streamed Research Plan
```

## Backend

`backend/app.py` exposes the main API:

- `GET /health` - backend and model readiness status
- `GET /config` - active model configuration
- `POST /search` - arXiv and Semantic Scholar collection plus hybrid retrieval
- `POST /summarize` - paper summary generation
- `POST /paper-guidance` - single-paper practical guidance
- `POST /research-plan` - non-streaming Research Plan generation
- `POST /research-plan/stream` - streaming Research Plan generation
- `POST /research-plan/cache-status` - cache compatibility endpoint
- `POST /paper-guidance/cache-status` - cache compatibility endpoint

## Retrieval Layer

The retrieval layer lives in `backend/retrieval/`.

- `bm25.py` searches exact terms across title, abstract, categories, and keywords.
- `dense.py` uses `BAAI/bge-small-en-v1.5` through `sentence-transformers` when available.
- `rrf.py` combines ranked lists with reciprocal rank fusion.
- `hybrid.py` runs BM25, dense search, and RRF together.
- `reranker.py` applies `BAAI/bge-reranker-base` as the final cross-encoder ranking step.

RRF is used because BM25 and dense search produce different score
distributions. It combines rankings without requiring score normalization.

## Model Layer

`backend/llm.py` wraps generation behind two providers:

- `transformers` - local PyTorch inference for AMD ROCm environments
- `fireworks` - optional cloud model path for deeper synthesis

Current default local model:

- `google/gemma-2-2b-it`

Task-specific model routing:

- `LOCAL_LLM_MODEL`
- `LOCAL_LLM_SUMMARY_MODEL`
- `LOCAL_LLM_GUIDANCE_MODEL`
- `LOCAL_LLM_RESEARCH_PLAN_MODEL`

## Frontend

`frontend/streamlit_app.py` provides the user workflow:

- search box for questions or paper links
- mutually exclusive Research Lens control
- ranked paper cards with category chips
- on-demand Summary and Guidance actions
- selected-paper Research Plan generation
- technical dashboard for retrieval and model details

## Deployment

The AMD notebook path runs the FastAPI backend and Streamlit frontend locally on
the VM, then exposes Streamlit through an Ngrok tunnel when the notebook
provider blocks Streamlit WebSockets through its native reverse proxy.

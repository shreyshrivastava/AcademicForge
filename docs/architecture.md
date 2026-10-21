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
  -> selected-paper Research Plan context
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
- `fireworks` - optional cloud model path for deeper synthesis when a Fireworks
  model ID and `FIREWORKS_API_KEY` are configured

Current default local model:

- `google/gemma-2-2b-it`

Task-specific model routing:

- `LOCAL_LLM_MODEL`
- `LOCAL_LLM_SUMMARY_MODEL`
- `LOCAL_LLM_GUIDANCE_MODEL`
- `LOCAL_LLM_RESEARCH_PLAN_MODEL`
- `LOCAL_LLM_DEEP_MODEL` for the optional Deep Mode model

Fast Mode uses the configured local model for summaries, guidance, and
Research Plans. Deep Mode is not a separate mandatory service: it only routes
to Fireworks when its configured model ID is a Fireworks model.

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
the VM. `start.sh` can expose Streamlit through an Ngrok tunnel when the
notebook provider blocks Streamlit WebSockets through its native reverse proxy.

## Deployment Profiles

| Capability | `amd-rocm-only` | `streamlit-cloud-demo` |
| --- | --- | --- |
| Runtime | FastAPI plus Streamlit on the AMD VM | One Streamlit process |
| Fast Mode | Local Gemma through PyTorch/ROCm | Lightweight demo generator |
| Deep Mode | Optional Fireworks model | Visible but disabled |
| Credentials | Hugging Face token for gated weights; Fireworks key only for Deep Mode | None |

The Cloud branch shares the same retrieval and user workflow, but it must not
be described as the AMD/Gemma runtime. Its in-process adapter is documented in
its own `docs/architecture.md`.

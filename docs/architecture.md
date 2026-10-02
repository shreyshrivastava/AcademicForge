# Architecture

AcademicForge has two processes:

- FastAPI backend in `backend/app.py`
- Streamlit frontend in `frontend/streamlit_app.py`

## Runtime Flow

```text
Streamlit UI
  -> FastAPI backend
  -> arXiv + Semantic Scholar live search
  -> relevance filtering
  -> BM25 lexical retrieval
  -> BGE dense retrieval
  -> Reciprocal Rank Fusion
  -> BGE cross-encoder reranking
  -> selected evidence papers
  -> local Gemma summaries and guidance
  -> compact Research Plan context
  -> local Gemma Research Plan, or opt-in Fireworks DeepSeek
```

## Backend Routes

- `GET /health`
- `GET /version`
- `GET /config`
- `POST /search`
- `POST /summarize`
- `POST /paper-guidance`
- `POST /research-plan`
- `POST /research-plan/stream`

## Model Routing

- `LOCAL_LLM_PROVIDER=auto` selects MLX on Apple Silicon and Transformers/PyTorch elsewhere.
- The Transformers/PyTorch backend supports both NVIDIA CUDA and AMD ROCm. The
  application uses the same code path for both because PyTorch exposes both
  accelerators through its CUDA device API.
- Fast local model on AMD/ROCm: `google/gemma-2-2b-it`.
- Fast local model on Apple Silicon: `mlx-community/gemma-2-2b-it-4bit`.
- Research Plan defaults to local Gemma.
- Fireworks DeepSeek is used only when `LOCAL_LLM_USE_FIREWORKS=true`.

## Retrieval

The retrieval layer lives in `backend/retrieval/`.

- `bm25.py` performs lexical keyword ranking.
- `dense.py` performs BGE dense semantic retrieval.
- `rrf.py` fuses BM25 and dense ranks.
- `reranker.py` applies a BGE cross-encoder.
- `device.py` selects GPU by default when PyTorch exposes one.

On ROCm, PyTorch exposes AMD GPUs through the `cuda` API, so `retrieval_device: cuda` in `/version` is expected. On NVIDIA, the same field indicates CUDA-backed retrieval.

## Frontend

The Streamlit UI provides:

- research question input
- Fast/Deep mode selector
- Research Lens selector
- ranked paper cards
- paper selection
- inline Summary and Guidance generation
- streamed Research Plan generation
- Markdown export

The UI talks to FastAPI through `ACADEMICFORGE_BACKEND_URL`.

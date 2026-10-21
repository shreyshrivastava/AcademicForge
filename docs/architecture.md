# Streamlit Cloud Demo Architecture

This branch keeps AcademicForge's research workflow and interface while using a
self-contained demo runtime suited to Streamlit Cloud. It is not the AMD ROCm
deployment; that implementation lives on `amd-rocm-only`.

## High-Level Flow

```text
Streamlit UI
  -> APIClient chooses an in-process demo adapter when no backend URL exists
  -> live source search
       -> arXiv
       -> Semantic Scholar
  -> candidate paper pool
  -> retrieval layer
       -> BM25 lexical ranking
       -> dense semantic ranking when dependencies are available
       -> reciprocal rank fusion
       -> cross-encoder reranking when dependencies are available
  -> relevance and Research Lens selection
  -> ranked paper cards
       -> on-demand Summary
       -> on-demand Guidance
       -> selected papers
  -> lightweight evidence-grounded Research Plan
```

## Deployment Adapter

`frontend/api_client.py` normally talks to FastAPI through
`ACADEMICFORGE_BACKEND_URL`. Streamlit Cloud hosts one Streamlit process, so
this branch switches to an in-process adapter when that variable is absent.

The adapter invokes the same product modules used by the server deployment:

- `backend.data_pipeline.retrieve_and_rank_papers()` for live search
- `backend.summarizer.summarize_paper()` for a paper summary
- `backend.guidance_generator.generate_paper_guidance()` for paper guidance
- `backend.research_plan_generator.stream_research_plan()` for the selected-paper plan

It presents the same logical endpoints to the UI, so the Streamlit interaction
model remains consistent with the AMD build without requiring an HTTP backend.

## Retrieval Layer

The retrieval modules are shared with the AMD build.

- `bm25.py` ranks exact language in titles, abstracts, categories, and tags.
- `dense.py` uses `BAAI/bge-small-en-v1.5` when `sentence-transformers` is
  installed, otherwise it uses a local lexical cosine fallback.
- `rrf.py` merges lexical and semantic rankings.
- `hybrid.py` coordinates BM25, dense ranking, and RRF.
- `reranker.py` uses `BAAI/bge-reranker-base` when it is installed and keeps
  the fused order if it cannot load.

Live sources are queried per request. Relevance checks and a domain mismatch
guard prevent unrelated papers from becoming synthesis input. Research Lens is
mutually exclusive: `Balanced` supplies no category filter; any other lens
supplies exactly one category preference.

## Generation Modes

### Fast Mode (Demo)

The demo provider turns retrieved title and abstract evidence into concise
summaries, project-oriented guidance, and structured Research Plans. It is a
lightweight public product demonstration and does not load a Gemma model.

### Deep Mode (Visible, disabled)

The UI retains Deep Mode to explain the full product experience. It is disabled
in this public branch because Fireworks credentials should not be embedded in a
public Streamlit deployment. The UI explicitly directs a user to the AMD build
for the optional DeepSeek/Fireworks route.

## What Is Different From the AMD Deployment

| Area | Streamlit Cloud demo | AMD deployment |
| --- | --- | --- |
| Process layout | One Streamlit process | Streamlit plus FastAPI |
| Fast inference | Lightweight demo provider | Local Gemma through PyTorch/ROCm |
| Deep inference | Disabled | Optional Fireworks/DeepSeek |
| Secrets | Not required | Hugging Face token; Fireworks key only for Deep Mode |
| Primary purpose | Public workflow demo | Full local-first AI research assistant |

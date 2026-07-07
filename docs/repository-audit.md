# Repository Audit

This audit summarizes the repository state after the hackathon-readiness pass.

## Current Strengths

- Clear FastAPI backend and Streamlit frontend separation.
- Modular retrieval layer under `backend/retrieval/`.
- Local-first LLM wrapper with MLX as the default provider.
- Summary and Research Plan generation are cached locally.
- Research Plan generation streams output to improve perceived latency.
- Tests cover retrieval, cache, model routing, generation pipeline, API contracts, and the benchmark helper.

## Issues Found

- README was useful for local development but not judge-ready.
- Architecture, setup, and product roadmap documentation were missing.
- `.env.example`, `LICENSE`, and `CONTRIBUTING.md` were missing.
- Generated Python cache files were present in source directories.
- Accidental empty directories existed from early scaffolding:
  - `-p/`
  - `frontendmkdir/`
  - `backend/frontend/`
- Early scaffold modules were stale and unreferenced:
  - `backend/prompts.py`
  - `backend/paper_search.py`

## Changes Made

- Rewrote `README.md` for hackathon/demo readability.
- Added:
  - `docs/architecture.md`
  - `docs/setup.md`
  - `docs/roadmap.md`
  - `docs/repository-audit.md`
  - `.env.example`
  - `CONTRIBUTING.md`
  - `LICENSE`
- Removed stale scaffold files and empty accidental directories.
- Cleaned generated `__pycache__` files from source folders.
- Confirmed `.gitignore` excludes local environment, cache, and generated Python artifacts.

## Recommended GitHub Metadata

Repository description:

```text
An AI-powered research-to-implementation engine that translates academic papers into production-ready system architectures and engineering blueprints.
```

Topics:

```text
academic-search
research-assistant
ai-search
hybrid-retrieval
bm25
dense-retrieval
reciprocal-rank-fusion
arxiv
fastapi
streamlit
mlx
local-llm
hackathon
```

## Remaining Presentation Work

- Add screenshots or a demo GIF under `docs/assets/`.
- Add a short demo video link to the README.
- Add one generated example Research Plan under `docs/examples/`.
- Add a repository social preview image.
- Add hosted demo instructions if deployment is completed.

## Intentional Non-Goals For Now

- No database.
- No user accounts.
- No hosted infrastructure requirement.
- No full-PDF ingestion.
- No active cross-encoder reranker yet.

These are reasonable post-hackathon improvements, but they are not necessary for a clear MVP submission.

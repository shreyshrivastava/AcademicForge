# Product roadmap

AcademicForge is currently a hackathon-ready MVP. The focus is clarity, local execution, and a strong research-to-build workflow.

## Completed

- arXiv and Semantic Scholar live search by topic or direct URL
- score-fusion retrieval: stemmed BM25 plus BGE dense retrieval
- optional RRF and cross-encoder retrieval modes
- paper selection UI
- local MLX summarization
- streamed local MLX Research Plan generation
- persistent summary and Research Plan cache
- numbered Research Plan citations tied to selected papers
- reproducible SciFact and frozen NFCorpus retrieval evaluations
- deterministic local Research Plan citation-membership and Key Findings citation-coverage screen

- lightweight tests for retrieval, generation, cache, and API contracts

## Near-Term

- Add screenshots and a short demo GIF.
- Add example generated Research Plan to `docs/examples/`.
- Improve paper cards with clearer badges for method, benchmark, theory, and dataset.
- Add a one-click "recommended selection" based on retrieval roles.
- Add better query spelling normalization.

## Mid-Term

- Add project and implementation discovery from GitHub.
- Add Papers with Code integration for code, datasets, and benchmarks.
- Add saved research sessions.
- Add comparison views for selected papers.
- Add richer export formats.

## Long-Term

- Add user workspaces.
- Add collaborative research boards.
- Add code-generation mode using a task-specific coder model.
- Compare local Gemma-2B with a larger model on the citation-faithfulness set.
- Evaluate future CUDA and ROCm inference backends when the MLX path is stable.

## Known limitations

- Search quality evidence covers two abstract-only BEIR datasets; it does not measure live-source recall, full texts, or user outcomes.
- Retrieval is in-memory and optimized for MVP candidate sets.
- Score fusion is the default; cross-encoder reranking remains opt-in because its SciFact gain was not significant.
- Local generation speed depends on hardware and selected model.
- Research Plans are generated from abstracts and summaries, not full PDFs. Citation IDs prove only that a cited paper was selected, not that each claim is entailed.

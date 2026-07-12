# Product Roadmap

AcademicForge is currently a hackathon-ready MVP. The focus is clarity, local execution, and a strong research-to-build workflow.

## Completed

- arXiv search by topic or direct URL
- BM25 keyword retrieval
- dense semantic retrieval
- reciprocal rank fusion
- cross-encoder reranking
- paper selection UI
- local Transformers summarization on the AMD/ROCm path
- streamed Research Plan generation
- arXiv and Semantic Scholar source search

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
- Add cross-encoder reranking.
- Add saved research sessions.
- Add comparison views for selected papers.
- Add richer export formats.

## Long-Term

- Add hosted demo deployment.
- Add user workspaces.
- Add collaborative research boards.
- Add code-generation mode using a task-specific coder model.
- Add evaluation harness for search relevance and Research Plan quality.
- Improve AMD/ROCm deployment automation and hosted demo reliability.

## Known Limitations

- Research Plans are generated from abstracts and metadata, not full PDFs.
- Retrieval is in-memory and optimized for MVP candidate sets.
- Local generation speed depends on hardware and selected model.

# Roadmap

AcademicForge is a working MVP built for a hackathon. The goal is a clear question-to-plan workflow that runs on your own machine.

## Done

- Live arXiv search by topic or arXiv ID, with Semantic Scholar as an optional second source when an API key is set
- Field presets and custom fields that separate meanings such as RAG and ragas
- Score-fusion ranking (stemmed BM25 + BGE), with optional RRF, cross-encoder and OpenJev
- Paper cards with per-paper summaries and guidance
- Streamed research plans with numbered citations to the selected papers
- Local generation on MLX, CUDA and ROCm, with optional Groq or Fireworks for Deep mode
- A redesigned single-column interface ([DESIGN.md](../DESIGN.md)) and plain-language prompts
- SciFact and frozen NFCorpus retrieval evaluations, a summary faithfulness screen and a plan citation check
- Screenshots and a worked example plan in [docs/examples](examples/)
- A standalone Streamlit demo that needs no keys

## Next

- Re-run the summary and plan checks with the current prompts.
- Add OpenAlex as another source, using its free API key so it isn't throttled.
- Compare Gemma 2B with a larger local model on the plan citation set.
- Show method, benchmark, theory and dataset badges on paper cards.
- A one-click recommended selection based on each paper's category.
- Better spelling correction for queries.

## Later

- Find code and datasets for a paper (GitHub, Papers with Code).
- Save research sessions and compare selected papers side by side.
- More export formats.
- Shared workspaces.

## Known limits

- The retrieval evidence covers two BEIR datasets of abstracts. It says nothing about live-search recall, full papers or whether users find the plans useful.
- Retrieval runs in memory and is sized for about a hundred candidates per search.
- Plans are written from abstracts and summaries, not full PDFs. A citation shows that a paper was selected, not that it supports the claim.
- Without a Semantic Scholar key, search covers arXiv only.
- Generation speed depends on your hardware and model.

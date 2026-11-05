# AcademicForge — Handoff (2026-10-29)

Repo: https://github.com/shreyshrivastava/AcademicForge (latest commit `401c24d`)
Code graph: `graphify-out/graph.html` (interactive), `graphify-out/GRAPH_REPORT.md`

## UPDATE 2026-11-05: most of steps 1–4 are already done on branch `benchmark-evidence`
The "findings" below describe `main` only. Branch `benchmark-evidence` (20 commits, not merged), plus `stash@{1}`, contains:
- Retrieval ablation on SciFact + NFCorpus. Default is now score fusion (stemmed BM25 + BGE, alpha 0.9). Reranker stays opt-in.
- Faithfulness screen with negative controls, a plan citation check, roadmap fix, and a GPU notebook.
- In stash only: 10-claim entailment pilot and an AI-assisted summary review pilot.

## What it is
Streamlit + FastAPI app: research question → arXiv/Semantic Scholar search → hybrid ranking (BM25 + BGE + RRF) → pick papers → summaries/guidance → streamed research plan (local Gemma-2-2B by default; MLX / CUDA / ROCm; optional cloud). Also a light Streamlit Cloud demo (BM25 + small CPU model).

## Decisions made in the grilling session
- Goal: defend the project to hackathon judges/interviewers and get strong, **real** CV numbers.
- Pitch: "evidence-grounded, actionable research plan"; local/private is supporting proof.
- Keep Gemma-2-2B as the local default; document its weaknesses.
- Keep all hardware paths; call MLX the tested "golden path", others best-effort.
- CI stays as-is (user decision). Benchmark remains a manual run.
- Numbers on the CV must come from published results in the repo, never extrapolated.

## Key findings
1. **Only BM25 is benchmarked.** SciFact nDCG@10 = 0.643 (a bit below BEIR's ~0.665 BM25). BGE, hybrid/RRF, reranker, and plan quality are unmeasured.
2. **CI does not run the benchmark** (`.github/workflows/ci.yml` = lint, compile, pytest only). Don't claim "benchmarks in CI".
3. **`docs/roadmap.md` is stale**: says arXiv-only and "hosted demo" is long-term; both are already done.
4. **No plan-quality check**: nothing verifies that plan citations exist in the selected papers.
5. Graph: hubs are `get_config()`, `retrieve_and_rank_papers()`, `enrich_papers_with_jev()`, `contextualize_query()`. `data_pipeline.py` (57 nodes, cohesion 0.05) and `llm.py` (cohesion 0.06) are loose grab-bags. No import cycles.
6. OpenJev (`backend/jev.py`) is an optional typed-decision layer; it needs a 15-second justification or removal.

## Steps to improve (in priority order)
1. **Run the retrieval ablation** with `benchmarks/run_scifact.py`: BM25 vs BGE vs hybrid (RRF) vs hybrid + reranker. Same protocol (300 queries, bootstrap CIs). Publish the table in `docs/benchmarks/README.md` and JSON next to `scifact-bm25.json`. This gives the headline CV number.
2. **Add a citation-faithfulness check** for research plans: every cited paper ID must be in the selection. Deterministic; add a test and report the % on a set of 10–20 questions.
3. **Fix `docs/roadmap.md`** to match what has shipped.
4. **Measure Gemma-2-2B vs one bigger model** (e.g. a 4B) on the step-2 set: faithfulness % and latency.
5. **Document failure behavior** (one source down / both down / rate-limited) in `docs/reliability.md` in plain terms.
6. **Split `data_pipeline.py`** into query expansion, field filtering, and dedupe modules; same for `llm.py` (provider selection vs generation).
7. **Justify or drop OpenJev**: one paragraph in `docs/architecture.md`.
8. Optional later: an LLM-as-judge groundedness rubric; a CI benchmark gate if you change your mind.

## CV bullet templates (fill only with measured numbers)
- Built hybrid BM25 + BGE retrieval with RRF; nDCG@10 0.64 → **[measured]** on BEIR SciFact (300 queries, 5.2k docs, 95% bootstrap CIs).
- Research plans with **[measured]%** citation faithfulness across **[N]** questions; local Gemma on Apple Silicon / CUDA / ROCm.
- FastAPI + Streamlit app with CI (lint, tests, hashed deps) and a public cost-limited demo.

## Next action
Say "run the ablation" to start step 1.

# AcademicForge

AcademicForge is an AI research-to-build workbench for finding, comparing, summarizing, and turning academic papers into implementation roadmaps.

Instead of returning only keyword matches, AcademicForge combines arXiv search, BM25 keyword retrieval, dense semantic retrieval, reciprocal rank fusion, local and cloud-ready LLM summarization, and streamed roadmap generation to help builders move from:

```text
What should I read?
```

to:

```text
What should I build next, and how?
```

## AMD Developer Challenge Positioning

AcademicForge is being built for the AMD Developer Challenge as a product-oriented AI application that connects research discovery with practical implementation planning.

The current project is intentionally split into two layers:

- **Product layer:** a usable research assistant that turns papers into summaries and build roadmaps.
- **Inference layer:** a provider-aware backend that can run locally with MLX today and move to AMD Cloud, ROCm, vLLM, Fireworks, or Gemma-hosted inference later.

This matters because MLX is excellent for fast local iteration on Apple Silicon, but AMD hardware is the natural path for a hosted, scalable, hackathon-ready deployment.

Current model strategy:

| Role | Current choice | Reason |
| --- | --- | --- |
| Best local default | `mlx-community/Qwen3-4B-4bit` | Best speed/quality balance in local tests. |
| Lightweight sponsor-aligned baseline | `mlx-community/gemma-3-1b-it-4bit` | Easy local smoke tests and routing experiments. |
| Gemma 4 local experiment | `mlx-community/gemma-4-e2b-it-OptiQ-4bit` | Works locally and gives clearer Gemma 4 output, but slower than Qwen on this Mac. |
| Future hosted model | Gemma 4 on Fireworks or AMD Cloud | Stronger sponsor/AMD story and better deployment fit. |

The product direction is not "one model wins forever." The stronger architecture is:

```text
Local development: MLX + Qwen for quality and fast iteration
Hackathon deployment: AMD Cloud / Fireworks + Gemma 4
Future optimization: router chooses the cheapest model that preserves answer quality
```

## What We Tested

Local tests were run on the AcademicForge summary and roadmap workflow, not generic chatbot prompts. The benchmark uses `scripts/benchmark_llm.py` and measures tiny generation, paper summaries, roadmap generation, output length, latency, and simple quality flags.

| Model | Local runtime | Total benchmark time | Tiny generation | Summary behavior | Roadmap behavior | Current verdict |
| --- | --- | ---: | ---: | --- | --- | --- |
| `mlx-community/gemma-3-1b-it-4bit` | MLX / `mlx-lm` | `26.399s` | `3.846s` | More verbose; sometimes invented details in earlier tests. | Usable but more generic. | Useful as a small router/smoke-test model, not the main demo model. |
| `mlx-community/Qwen3-4B-4bit` | MLX / `mlx-lm` | `23.870s` to `35.890s` | `1.595s` to `1.787s` | Concise, cleaner, more faithful to abstracts. | Best speed/quality balance for AcademicForge today. | Default local model. |
| `mlx-community/gemma-4-e2b-it-4bit` | MLX / `mlx-lm` and `mlx-vlm` | Did not complete | n/a | Failed local load with a Gemma 4 weight mismatch. | n/a | Not used. |
| `mlx-community/gemma-4-e2b-it-OptiQ-4bit` | MLX / `mlx-lm` | `49.140s` | `4.607s` | Good and clear; more detailed than Gemma 3. | Stronger than Gemma 3 but slower and more verbose than Qwen. | Keep as Gemma 4 compatibility proof and future AMD/Fireworks candidate. |

Benchmark command:

```bash
python scripts/benchmark_llm.py \
  --model mlx-community/gemma-4-e2b-it-OptiQ-4bit,mlx-community/Qwen3-4B-4bit \
  --json-out benchmark-results.json \
  --markdown-out benchmark-results.md
```

Key local conclusion:

```text
Qwen is the best local default for the current user experience.
Gemma 4 is strategically important for AMD/Gemma deployment and sponsor alignment.
```

## AMD Hardware Advantage

The local Mac tests are useful for development, but they are not the final deployment target. AMD hardware can make AcademicForge a stronger product in three ways:

1. **Hosted inference:** run a larger Gemma or Qwen model on AMD Cloud instead of relying on local MLX.
2. **Benchmark credibility:** compare latency, throughput, and output quality between local MLX and AMD-hosted inference.
3. **Router/fine-tuning path:** train or tune a routing/evaluation layer that decides when a small model is enough and when a larger hosted model is worth the cost.

Planned AMD comparison:

| Experiment | Local baseline | AMD Cloud / Fireworks target | What we measure |
| --- | --- | --- | --- |
| Paper summary | Qwen3 4B MLX | Gemma 4 or Qwen on ROCm/vLLM | Latency, faithfulness, concision. |
| Roadmap generation | Qwen3 4B MLX | Gemma 4 on AMD/Fireworks | Completeness, implementation usefulness, hallucination rate. |
| Router decision | Gemma 3 1B MLX | Hosted larger model fallback | Token cost saved without quality loss. |
| Batch benchmarking | Local single-user run | AMD GPU endpoint | Throughput and scalability. |

## Fine-Tuning And Routing Roadmap

Fine-tuning is not required for the current MVP, but it is an important next step for making AcademicForge feel more specialized than a generic chatbot.

Near-term tuning plan:

1. Collect AcademicForge examples: paper abstract, expected summary, expected roadmap, and quality labels.
2. Create a small evaluation set for summary faithfulness and roadmap usefulness.
3. Fine-tune or prompt-tune a small router/evaluator model to decide:
   - summarize locally
   - roadmap locally
   - call hosted Gemma 4 / AMD model
   - ask the user for a narrower goal
4. Compare prompt-only routing vs fine-tuned routing using the same benchmark harness.

The intended advantage is not just better generation. It is token-efficient routing:

```text
Use the cheapest model that can solve the task well.
Escalate only when the task requires stronger reasoning.
```

## Problem

Researchers, students, and hackathon builders often lose time moving between search engines, PDFs, paper abstracts, project ideas, and implementation plans. Normal search is useful, but it usually:

- over-ranks exact keyword matches and misses semantically relevant papers
- gives little help comparing why papers matter
- does not separate methods, benchmarks, theory, datasets, and implementation ideas
- leaves users to manually synthesize a roadmap from several papers

AcademicForge focuses on the early research-to-build workflow: discover candidates, compare them, select the best papers, summarize them, and generate a practical implementation direction.

## Solution

AcademicForge uses a hybrid retrieval and local generation pipeline:

1. Search arXiv for candidate papers.
2. Rank candidates with BM25 for exact technical terms.
3. Rank candidates with dense semantic search for meaning-level similarity.
4. Fuse rankings with reciprocal rank fusion.
5. Re-rank the top evidence pool.
6. Select a balanced 8-10 paper evidence set.
7. Let the user inspect and select papers.
8. Summarize selected papers with a local MLX model.
9. Generate streamed research synthesis from compact evidence notes.
10. Cache summaries and synthesis outputs locally for fast repeat runs.

## Evidence Selection Strategy

AcademicForge does not send every retrieved paper to the model. The retrieval funnel is:

```text
Retrieve 20-30 papers
  -> re-rank with hybrid retrieval signals
  -> select the best 8-10 evidence papers
  -> synthesize for the user's specific goal
```

The selected evidence set aims for this balance:

| Evidence type | Target count | Why it matters |
| --- | ---: | --- |
| Foundational papers | 2-3 | Gives the synthesis stable concepts and established methods. |
| Recent papers | 2-3 | Keeps the answer current and aligned with newer work. |
| Implementation-focused papers | 2-3 | Helps convert research into buildable systems. |
| Evaluation-focused papers | 1-2 | Provides benchmarks, datasets, or metrics. |
| Contrarian or alternative papers | 1-2 | Surfaces limitations, tradeoffs, and non-obvious directions. |

The current selector uses transparent heuristics from titles, abstracts, and dates.
Future versions can replace this with a learned router or cross-encoder reranker.

## Key Features

- arXiv search by topic or direct arXiv URL.
- Hybrid retrieval: BM25 + dense embeddings + reciprocal rank fusion.
- Optional reranker placeholder for future cross-encoder reranking.
- Paper selection before expensive LLM calls.
- Local MLX summarization and roadmap generation.
- Task-specific model routing for summaries and roadmaps.
- Streamed roadmap output so users see progress quickly.
- Persistent disk cache for summaries and roadmaps.
- Streamlit UI for hackathon-friendly demos.
- FastAPI backend with testable retrieval and generation layers.

## Architecture

```text
User query
  -> FastAPI /search
  -> arXiv candidate collection
  -> BM25 ranking
  -> dense semantic ranking
  -> reciprocal rank fusion
  -> optional reranker placeholder
  -> Streamlit paper comparison UI
  -> selected papers
  -> local MLX summaries
  -> compact roadmap context
  -> streamed local MLX roadmap
  -> local cache
```

More detail: [docs/architecture.md](docs/architecture.md)

## Technology Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI
- **Paper source:** arXiv API
- **Keyword retrieval:** in-memory BM25
- **Dense retrieval:** `sentence-transformers` with `BAAI/bge-small-en-v1.5` when available
- **Rank fusion:** reciprocal rank fusion
- **Local LLM runtime:** MLX via `mlx-lm`
- **Default local model:** `mlx-community/Qwen3-4B-4bit`
- **Caching:** local JSON cache under `.academicforge_cache/`

## Repository Structure

```text
AcademicForge/
  backend/
    app.py                  # FastAPI routes
    cache.py                # local disk cache helpers
    llm.py                  # local LLM provider wrapper
    retrieval/              # BM25, dense search, RRF, hybrid search
    roadmap_generator.py    # roadmap prompt, cache, streaming generation
    summarizer.py           # paper summarization and cache
  frontend/
    streamlit_app.py        # Streamlit UI
  scripts/
    benchmark_llm.py        # local model benchmark utility
    run_local.py            # one-command backend/frontend runner
  tests/
    test_*.py               # lightweight script-style tests
  docs/
    architecture.md
    setup.md
    roadmap.md
```

## Installation

Recommended: Python 3.11+ on macOS with Apple Silicon for MLX.

```bash
git clone https://github.com/shreyshrivastava/AcademicForge.git
cd AcademicForge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Optional dense retrieval dependency:

```bash
pip install sentence-transformers
```

Full setup guide: [docs/setup.md](docs/setup.md)

## Future GPU Backends

AcademicForge uses a provider-aware LLM service in `backend/llm.py`. MLX is the default and tested local runtime. `cuda`, `rocm`, `roc`, `torch`, and `transformers` are accepted as provider aliases for the Transformers backend, so the same app surface can run in future GPU-capable environments without changing the API or frontend.

On this Mac, keep using MLX. In an AMD/ROCm environment, install a compatible PyTorch/Transformers stack and set:

```bash
export LOCAL_LLM_PROVIDER=rocm
export LOCAL_LLM_MODEL=<a-compatible-hugging-face-causal-lm>
```

For a hosted hackathon demo, the preferred path is to run the UI/API as the product surface and connect the generation layer to an AMD Cloud or Fireworks-hosted model endpoint. Local MLX remains the development baseline; AMD Cloud becomes the shipping and benchmarking target.

## Configuration

The app is local-first. By default it uses MLX and Qwen:

```bash
export LOCAL_LLM_PROVIDER=mlx
export LOCAL_LLM_MODEL=mlx-community/Qwen3-4B-4bit
export LOCAL_LLM_SUMMARY_MODEL=mlx-community/Qwen3-4B-4bit
export LOCAL_LLM_ROADMAP_MODEL=mlx-community/Qwen3-4B-4bit
```

Useful environment variables:

```bash
LOCAL_LLM_PROVIDER=mlx
# LLM_BACKEND is also accepted as a compatibility alias.
# Accepted provider values: mlx, transformers, torch, cuda, rocm, roc
LOCAL_LLM_MODEL=mlx-community/Qwen3-4B-4bit
LOCAL_LLM_SUMMARY_MODEL=mlx-community/Qwen3-4B-4bit
LOCAL_LLM_ROADMAP_MODEL=mlx-community/Qwen3-4B-4bit
LOCAL_LLM_MAX_TOKENS=900
LOCAL_LLM_TEMPERATURE=0.2
ACADEMICFORGE_CACHE_DIR=.academicforge_cache
ACADEMICFORGE_CACHE_TTL_SECONDS=604800
ACADEMICFORGE_CACHE_MAX_FILES=500
ACADEMICFORGE_BACKEND_URL=http://localhost:8000
```

## Running Locally

The easiest local runner starts both FastAPI and Streamlit with the selected model:

```bash
source venv/bin/activate
python scripts/run_local.py --model mlx-community/Qwen3-4B-4bit --reload
```

Then open:

```text
http://127.0.0.1:8501
```

Manual backend/frontend startup is also supported.

Start the backend:

```bash
source venv/bin/activate
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Start the frontend in another terminal:

```bash
source venv/bin/activate
streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

## Usage

1. Enter a research query, such as `reduce ai hallucination`.
2. Review the ranked paper candidates.
3. Inspect BM25 rank, dense rank, RRF score, source, authors, URL, and abstract snippet.
4. Select 2-3 papers for a focused roadmap.
5. Generate summaries and a streamed implementation roadmap.
6. Download the roadmap as Markdown.

## Demo Script

Suggested live demo query:

```text
reduce ai hallucination
```

Recommended paper selection for a text/RAG hallucination roadmap:

- `First Hallucination Tokens Are Different from Conditional Ones`
- `OpenHalDet: A Unified Benchmark for Hallucination Detection across Diverse Generation Scenarios`
- `Probabilistic distances-based hallucination detection in LLMs with RAG`

What to point out to judges:

- Hybrid retrieval surfaces papers that keyword-only search may miss.
- Dense and BM25 ranks are visible for explainability.
- Users choose papers before LLM generation.
- Roadmap output streams from a local MLX model.
- Repeated summaries and roadmaps load instantly from cache.

## Evaluation

Current validation includes:

- BM25 returns relevant ranked results.
- Dense search returns ranked results and falls back cleanly if embeddings are unavailable.
- RRF deduplicates papers and promotes overlap between rankers.
- Summary and roadmap caches reuse previous generations.
- Roadmap generation uses compact paper notes.
- Streaming roadmap endpoint returns incremental text.
- FastAPI contract tests cover config, cache status, and streaming routes.

Run checks:

```bash
source venv/bin/activate
python -m py_compile backend/app.py backend/cache.py backend/llm.py backend/summarizer.py backend/roadmap_generator.py frontend/streamlit_app.py backend/retrieval/*.py scripts/*.py tests/*.py
python tests/test_cache.py
python tests/test_generation_pipeline.py
python tests/test_llm_routing.py
python tests/test_api_contract.py
python tests/test_benchmark_script.py
python tests/test_retrieval.py
```

## Local Model Benchmarking

Run the app with a specific local model:

```bash
python scripts/run_local.py --model mlx-community/Qwen3-4B-4bit --reload
```

Run only the backend, useful when Streamlit is already open:

```bash
python scripts/run_local.py --backend-only --model mlx-community/Qwen3-4B-4bit --reload
```

Benchmark the current local MLX setup:

```bash
python scripts/benchmark_llm.py --skip-summary
```

Benchmark Gemma 4 against Qwen:

```bash
python scripts/benchmark_llm.py \
  --model mlx-community/gemma-4-e2b-it-OptiQ-4bit,mlx-community/Qwen3-4B-4bit
```

Benchmark multiple models and write reports:

```bash
python scripts/benchmark_llm.py \
  --model mlx-community/gemma-3-1b-it-4bit,mlx-community/Qwen3-4B-4bit \
  --runs 2 \
  --skip-summary \
  --json-out benchmark-results.json \
  --markdown-out benchmark-results.md
```

For timing comparisons, the benchmark isolates its cache by default. Add
`--use-cache` when you want to measure warm-cache app behavior.

Earlier local benchmark notes:

| Model | Tiny generation | Roadmap generation | Notes |
| --- | ---: | ---: | --- |
| `mlx-community/Qwen3-4B-4bit` | 3.41s | 13.11s | Best current balance of speed and instruction following. |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 4.12s | 10.67s | Faster roadmap generation, but less precise on implementation/runtime details. |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | 3.36s | 30.92s | Stronger writing and scope control, but too slow for current UX. |

Gemma 4 local note:

```bash
pip install -U mlx-vlm
```

The standard `mlx-community/gemma-4-e2b-it-4bit` conversion failed locally with a weight-loading mismatch. The OptiQ text-generation build worked:

```bash
python scripts/run_local.py --model mlx-community/gemma-4-e2b-it-OptiQ-4bit --reload
```

## Limitations

- arXiv is currently the only paper source.
- Dense retrieval is in-memory and intended for MVP-scale candidate sets.
- Reranking is a placeholder; no cross-encoder reranker is enabled yet.
- Local MLX generation speed depends on hardware and model size.
- The roadmap is only as good as the selected papers and available abstracts.
- No user accounts, saved projects, or hosted deployment are included yet.

## Future Work

- Add GitHub repository/project discovery.
- Add dataset discovery from Papers with Code, Hugging Face, or Kaggle.
- Add cross-encoder reranking.
- Add saved research sessions.
- Add export bundles for hackathon implementation plans.
- Add code-generation mode with a coder model separate from the roadmap model.
- Add hosted demo instructions or a lightweight Docker path.

More detail: [docs/roadmap.md](docs/roadmap.md)

## GitHub Repository Recommendations

Suggested repository description:

```text
AI-powered academic discovery platform with hybrid retrieval, local MLX summaries, and streamed research roadmaps.
```

Suggested topics:

```text
academic-search, research-assistant, ai-search, hybrid-retrieval, bm25, dense-retrieval, reciprocal-rank-fusion, arxiv, fastapi, streamlit, mlx, local-llm, hackathon
```

Recommended assets to add before final submission:

- short demo GIF or screenshots in `docs/assets/`
- 2-minute demo video link
- architecture diagram image
- example generated roadmap

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE).

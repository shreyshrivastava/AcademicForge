# AcademicForge

AcademicForge is a local-first AI research discovery platform for finding, comparing, and turning academic papers into implementation roadmaps.

Instead of returning only keyword matches, AcademicForge combines arXiv search, BM25 keyword retrieval, dense semantic retrieval, reciprocal rank fusion, local LLM summarization, and streamed roadmap generation to help builders move from "what should I read?" to "what should I build next?"

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
5. Let the user inspect and select papers.
6. Summarize selected papers with a local MLX model.
7. Generate a streamed implementation roadmap from compact paper notes.
8. Cache summaries and roadmaps locally for fast repeat runs.

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
- **Local LLM runtime:** MLX via `mlx-lm`; optional Transformers backend for NVIDIA CUDA or AMD ROCm
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

## Optional NVIDIA CUDA or AMD ROCm Setup

AcademicForge defaults to MLX because this repository was developed on Apple Silicon. If you want to run the local LLM on an NVIDIA or AMD machine, use the Transformers backend instead.

Install the base dependencies first:

```bash
pip install -r requirements.txt
pip install transformers accelerate sentencepiece safetensors
```

Then install the correct PyTorch build for your hardware.

NVIDIA CUDA example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

AMD ROCm example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Then configure a Hugging Face causal language model:

```bash
export LOCAL_LLM_PROVIDER=transformers
export LOCAL_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
export LOCAL_LLM_SUMMARY_MODEL=Qwen/Qwen2.5-3B-Instruct
export LOCAL_LLM_ROADMAP_MODEL=Qwen/Qwen2.5-3B-Instruct
```

For AMD ROCm you can also use:

```bash
export LOCAL_LLM_PROVIDER=rocm
```

`rocm` currently routes through the same Transformers code path as `transformers`; the difference is the PyTorch build and GPU driver/runtime installed on the machine.

Use the official PyTorch install selector for the latest CUDA/ROCm wheel command: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). AMD also publishes ROCm PyTorch installation notes: [ROCm PyTorch install guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html).

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
LOCAL_LLM_MODEL=mlx-community/Qwen3-4B-4bit
LOCAL_LLM_SUMMARY_MODEL=mlx-community/Qwen3-4B-4bit
LOCAL_LLM_ROADMAP_MODEL=mlx-community/Qwen3-4B-4bit
LOCAL_LLM_MAX_TOKENS=900
LOCAL_LLM_TEMPERATURE=0.2
ACADEMICFORGE_CACHE_DIR=.academicforge_cache
ACADEMICFORGE_BACKEND_URL=http://localhost:8000
```

## Running Locally

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

Open:

```text
http://127.0.0.1:8501
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

Benchmark the current local MLX setup:

```bash
python scripts/benchmark_llm.py --skip-summary
```

Benchmark another MLX model:

```bash
python scripts/benchmark_llm.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --skip-summary
```

Recent local benchmark notes:

| Model | Tiny generation | Roadmap generation | Notes |
| --- | ---: | ---: | --- |
| `mlx-community/Qwen3-4B-4bit` | 3.41s | 13.11s | Best current balance of speed and instruction following. |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | 4.12s | 10.67s | Faster roadmap generation, but less precise on implementation/runtime details. |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | 3.36s | 30.92s | Stronger writing and scope control, but too slow for current UX. |

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

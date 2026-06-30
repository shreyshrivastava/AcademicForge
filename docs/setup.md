# Setup Guide

This guide gets AcademicForge running locally.

## Requirements

- Python 3.11+
- macOS with Apple Silicon recommended for MLX
- Internet access for arXiv search and initial model downloads

## 1. Create Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional dense retrieval dependency:

```bash
pip install sentence-transformers
```

## 2. Configure Environment

```bash
cp .env.example .env
```

Default local model settings:

```bash
export LOCAL_LLM_PROVIDER=mlx
export LOCAL_LLM_MODEL=mlx-community/Qwen3-4B-4bit
export LOCAL_LLM_SUMMARY_MODEL=mlx-community/Qwen3-4B-4bit
export LOCAL_LLM_ROADMAP_MODEL=mlx-community/Qwen3-4B-4bit
```

## 3. Start Backend

```bash
source venv/bin/activate
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Check backend health:

```bash
curl http://127.0.0.1:8000/config
```

## 4. Start Frontend

In a second terminal:

```bash
source venv/bin/activate
streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

## 5. Run Tests

```bash
python -m py_compile backend/app.py backend/cache.py backend/llm.py backend/summarizer.py backend/roadmap_generator.py frontend/streamlit_app.py backend/retrieval/*.py scripts/*.py tests/*.py
python tests/test_cache.py
python tests/test_generation_pipeline.py
python tests/test_llm_routing.py
python tests/test_api_contract.py
python tests/test_benchmark_script.py
python tests/test_retrieval.py
```

## Troubleshooting

### Backend is not running

Start it:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend cannot reach backend

Set the backend URL:

```bash
export ACADEMICFORGE_BACKEND_URL=http://127.0.0.1:8000
```

### First generation is slow

The first run may download or load local MLX model weights. Repeated summaries and roadmaps are cached.

### Dense retrieval model warning

If `sentence-transformers` is not installed, dense retrieval falls back to a lexical cosine scorer. Install it for BGE embeddings:

```bash
pip install sentence-transformers
```

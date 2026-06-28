# AcademicForge

AcademicForge is a local-first research assistant that searches arXiv, summarizes papers with a local LLM, and generates implementation roadmaps.

## Features

- Search arXiv by research query or direct arXiv URL.
- Summarize paper abstracts with a local model.
- Generate practical implementation steps from the selected papers.
- Export the generated research roadmap as Markdown.
- Uses an interchangeable local LLM provider layer, with MLX as the current default.

## Current Local Model Setup

The app currently runs with MLX:

```bash
export LOCAL_LLM_PROVIDER=mlx
export LOCAL_LLM_MODEL=mlx-community/Qwen3-4B-4bit
```

The provider layer is in `backend/llm.py`, so future AMD/ROCm support can be added behind the same interface.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Start the backend:

```bash
source venv/bin/activate
export LOCAL_LLM_PROVIDER=mlx
export LOCAL_LLM_MODEL=mlx-community/Qwen3-4B-4bit
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend:

```bash
source venv/bin/activate
streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

Example direct run:

```text
http://127.0.0.1:8501/?query=https%3A%2F%2Farxiv.org%2Fabs%2F1706.03762&run=1
```

## Notes

- `.env`, virtual environments, and Python cache files are ignored.
- Local generation can take time depending on the model and hardware.

# Contributing

Thanks for helping improve AcademicForge.

## Development Principles

- Keep the project local-first and easy to run.
- Prefer small, understandable changes over broad rewrites.
- Preserve the existing FastAPI + Streamlit workflow unless there is a clear reason to change it.
- Add tests for retrieval, cache, API, or generation behavior when changing those areas.
- Avoid adding databases or hosted services unless they are necessary for the feature.

## Local Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-local.txt
cp .env.example .env
```

Run backend:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Run frontend:

```bash
streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

## Test Before Submitting

```bash
python -m py_compile backend/app.py backend/cache.py backend/llm.py backend/summarizer.py backend/guidance_generator.py backend/research_plan_generator.py frontend/streamlit_app.py backend/retrieval/*.py scripts/*.py tests/*.py
python tests/test_cache.py
python tests/test_generation_pipeline.py
python tests/test_llm_routing.py
python tests/test_api_contract.py
python tests/test_retrieval.py
```

## Pull Request Checklist

- Explain the problem and the change.
- Include screenshots or demo notes for UI changes.
- Include test output.
- Keep unrelated refactors out of the PR.
- Update docs when behavior changes.

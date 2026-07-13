# Contributing

Keep changes small, testable, and aligned with the current FastAPI + Streamlit architecture.

## Local Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --ignore-installed blinker -r requirements-local.txt
cp .env.example .env
```

## Run

```bash
./start.sh
```

## Test Before Submitting

```bash
python -m py_compile backend/*.py backend/retrieval/*.py frontend/*.py scripts/*.py tests/*.py
python tests/test_generation_pipeline.py
python tests/test_llm_routing.py
python tests/test_retrieval_device.py
python tests/test_api_contract.py
python tests/test_retrieval.py
```

## Checklist

- Explain the problem and the change.
- Keep unrelated refactors out.
- Update docs when behavior changes.
- Include screenshots or demo notes for UI changes.
- Include the test output you actually ran.

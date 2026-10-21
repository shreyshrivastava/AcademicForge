# Contributing

Keep changes focused and preserve the FastAPI/Streamlit boundaries. Use Python 3.12–3.14.

## Setup

```bash
uv sync --locked --extra local
cp .env.example .env
source .venv/bin/activate
./start.sh
```

Use a Hugging Face token only if your model needs gated access. Cloud generation requires explicit opt-in. GPU VMs use [the dedicated setup path](docs/setup.md#gpu-vms).

## Checks

```bash
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check backend frontend scripts tests benchmarks
uv run --no-sync python -m compileall -q backend frontend scripts tests benchmarks
bash -n start.sh
uv pip check
```

Unit tests isolate configuration and use fake models/providers. Add failure tests when changing source retrieval, retries, model routing, startup, admission control, or streams. Keep hardware/network checks separate from unit tests. Never call paid providers or load large models implicitly in a unit test.

Update dependencies in `pyproject.toml`, run `uv lock`, and regenerate exports with `python -m scripts.export_requirements`. Do not hand-edit generated package pins. Include the exact validation performed and screenshots for UI changes in a pull request.

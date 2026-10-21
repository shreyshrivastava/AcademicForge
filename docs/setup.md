# Setup

AcademicForge runs locally or on a GPU VM. Use Python 3.12–3.14. Deployment uses the same FastAPI and Streamlit processes on every platform.

## Local installation

```bash
git clone https://github.com/shreyshrivastava/AcademicForge.git
cd AcademicForge
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
cp .env.example .env
./start.sh
```

Alternatively, with uv installed, `uv sync --locked --extra local` creates the locked development environment. Activate `.venv/bin/activate` before launching.

The example configuration leaves the model unset. `auto` selects MLX with `mlx-community/gemma-2-2b-it-4bit` on Apple Silicon and Transformers with `google/gemma-2-2b-it` elsewhere. For gated models, accept access terms on Hugging Face and set `HF_TOKEN` in `.env`.

The app opens at [localhost:8501](http://127.0.0.1:8501). First startup downloads weights. It has a configurable 600-second readiness deadline and cleans up its child processes if startup or either service fails. Increase the deadline for a slow initial download, or cache weights explicitly with `python -m scripts.download_weights`.

## GPU VMs

Start with a VM image that already has a verified **PyTorch 2.12.x** build for its CUDA or ROCm runtime. Follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for compatible hardware, Python, drivers, and wheels. This repository does not choose a GPU wheel by adding an unrelated package index.

After activating an environment containing that stack:

```bash
# Choose one:
python -m scripts.install_gpu cuda
python -m scripts.install_gpu rocm
```

The helper checks the requested accelerator, installs the locked application packages with `--no-deps`, runs dependency validation, and checks the GPU again. It does not replace PyTorch, CUDA/ROCm libraries, or Triton. An incompatible image fails with an actionable error before application installation. The retained compatibility files `requirements-cuda.txt` and `requirements-rocm.txt` point to the shared GPU application export; prefer the installer above.

Set `LOCAL_LLM_PROVIDER=transformers` in `.env`. GPU builds are deployment-specific: the unit suite is not a certification of any VM image. Record the exact image, driver, PyTorch wheel, model revision, `/version` response, and a successful search plus generation before deploying that image.

## Network access

The launcher binds both processes to loopback by default. For an authenticated backend exposed through a VM proxy, set these in `.env`:

```dotenv
ACADEMICFORGE_BIND_HOST=0.0.0.0
ACADEMICFORGE_API_KEY=replace-with-a-long-random-key
```

The Streamlit client sends the API key in `X-API-Key`. Backend routes other than `/live` and `/health` require it when configured. For public/shared access, also put the **Streamlit interface** behind a login-protected HTTPS gateway: the backend key protects backend calls, not access to the UI itself. Restrict direct access to backend ports at the firewall.

On the Radeon notebook VM used for earlier demos, the app proxy path is `/instances/<instance-id>/proxy/8501/`; backend diagnostics are under `/instances/<instance-id>/proxy/8000/version`. Direct diagnostic calls need the API-key header. An SSH tunnel is another option for keeping services private.

## Readiness and failures

- `/live` responds without model checks.
- `/health` reports `generation`, `dense`, and `reranker` readiness. It returns HTTP 503 until generation is ready, or if generation cannot load.
- If generation loads but optional ranking models fail, readiness is `degraded`; search can use lexical scoring and preserve the previous ranking.
- Search reports per-source status and partial-result warnings. Both sources failing produces HTTP 503 rather than “no evidence.”
- Generation is limited to one concurrent request by default; excess work gets HTTP 429. Cancelling a client does not free a slot until the underlying model operation stops.
- Cloud generation has explicit opt-in and a bounded SDK timeout. A pre-output cloud failure can fall back locally. Failure after partial stream output ends the attempt instead of appending a different answer.
- The frontend never automatically replays generation POSTs. Restart explicitly after an interrupted attempt.
- Request bodies are limited to 1 MiB and 60 requests per minute per client IP per process by default. Multi-process deployments need shared gateway limits.

FastAPI/Streamlit have no managed-host deployment path in this project. Use local startup or a supervised VM deployment.

## Development and dependency updates

```bash
uv sync --locked --extra local
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check backend frontend scripts tests benchmarks
uv run --no-sync python -m compileall -q backend frontend scripts tests benchmarks
bash -n start.sh
```

Update `pyproject.toml` intentionally, run `uv lock`, then regenerate pip exports with `python -m scripts.export_requirements`. Commit the lock and exports together. CI checks Linux/macOS and Python 3.12/3.14; unit tests use no developer `.env` or real generation providers.

A narrow MLX import guard skips only the legacy `NewlineTokenizer` registration in the pinned mlx-lm release, then restores `AutoTokenizer.register`. It does not modify Transformers private classes globally. Remove this compatibility module after validating an upstream mlx-lm release that fixes the registration.

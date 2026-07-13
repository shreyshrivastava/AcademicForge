# Setup

This is the current terminal-only setup path for AcademicForge.

## Local Or VM Install

```bash
git clone https://github.com/shreyshrivastava/AcademicForge.git
cd AcademicForge

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --ignore-installed blinker -r requirements-local.txt

cp .env.example .env
```

Edit `.env` and add real tokens if needed.

## Required Keys

- `HF_TOKEN`: needed for gated Hugging Face models such as Gemma.
- `FIREWORKS_API_KEY`: optional, but recommended. When present, Research Plan generation uses Fireworks DeepSeek.

## Run

```bash
./start.sh
```

Open:

```text
http://127.0.0.1:8501
```

Check diagnostics:

```bash
curl http://127.0.0.1:8000/version
```

## AMD ROCm VM Notes

Some notebook VMs have certificate issues with Git. Use:

```bash
git -c http.sslVerify=false clone https://github.com/shreyshrivastava/AcademicForge.git
git -c http.sslVerify=false pull origin main
```

Open the Streamlit app through the VM proxy:

```text
https://radeon-global.anruicloud.com/instances/<instance-id>/proxy/8501/
```

Open backend diagnostics through:

```text
https://radeon-global.anruicloud.com/instances/<instance-id>/proxy/8000/version
```

Do not use `VERCEL_API_URL` or `/spaces/...` URLs.

## Current Routing

- Fast Mode summaries and guidance use local Gemma.
- Research Plan uses Fireworks DeepSeek when `FIREWORKS_API_KEY` is present.
- Research Plan falls back to local Gemma when no Fireworks key exists.
- Dense retrieval and reranking use GPU by default when PyTorch exposes one.
- Set `ACADEMICFORGE_RETRIEVAL_DEVICE=cpu` to force retrieval to CPU.

## Troubleshooting

If ports are busy:

```bash
pkill -f streamlit || true
pkill -f uvicorn || true
./start.sh
```

If install fails on the VM with a `blinker` uninstall error, use the install command shown above:

```bash
python -m pip install --ignore-installed blinker -r requirements-local.txt
```

If startup is slow the first time, it is usually downloading or preloading model weights. Later starts should be faster because the Hugging Face cache is warm.

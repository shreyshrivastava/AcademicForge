# AMD ROCm Notebook Setup Guide

This guide runs AcademicForge on an AMD ROCm notebook VM and exposes the
Streamlit UI through Ngrok. Do not use the notebook provider's built-in
Streamlit/Hugging Face UI proxy for the app UI; Streamlit relies on WebSockets,
which notebook reverse proxies often break.

## What Runs Where

- FastAPI backend: `127.0.0.1:8000` inside the notebook VM
- Streamlit frontend: `127.0.0.1:8501` inside the notebook VM
- Public UI: `https://...ngrok-free.app` printed by `start.sh`

`HF_TOKEN` is only used for downloading Hugging Face model weights. It is not
used to host the UI.

## Run In The Notebook Terminal

```bash
cd /workspace/AcademicForge
source venv/bin/activate

export HF_TOKEN="your_huggingface_read_token"
export NGROK_TOKEN="your_ngrok_token"
export FIREWORKS_API_KEY="your_fireworks_key"  # optional for Deep Mode

./start.sh
```

When the script prints:

```text
OPEN THIS PUBLIC URL IN YOUR BROWSER:
https://...ngrok-free.app
```

open that Ngrok URL from your laptop browser.

Do not open:

```text
http://localhost:8501
```

from your laptop. That URL points to your laptop, not the remote AMD notebook VM.

## Quick Health Checks

If Ngrok fails, first verify the local services inside the notebook VM:

```bash
curl http://127.0.0.1:8000/health
curl -I http://127.0.0.1:8501
```

Both must respond before Ngrok can expose the UI.

## Notes

- The app unsets notebook/Spaces path variables before starting Streamlit.
- Streamlit is bound to IPv4 `127.0.0.1` to avoid IPv6 `localhost` mismatch.
- Ngrok connects directly to `127.0.0.1:8501`.

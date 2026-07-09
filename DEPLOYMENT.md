# AcademicForge Deployment Guide

AcademicForge is designed as a fully local architecture. To expose the local API backend so your Streamlit frontend can securely access it over the public internet, we recommend using **Cloudflare Tunnels**.

## Architecture
`Streamlit Frontend` → `Cloudflare Tunnel (Public URL)` → `Local FastAPI Backend (Mac/AMD)`

## 1. Run the Local Backend
Start the FastAPI backend on your local machine (e.g., port 8000).

```bash
# Export your desired ML provider (mlx or transformers)
export LOCAL_LLM_PROVIDER="mlx"

# Start the server
./venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## 2. Expose with Cloudflare Tunnel
Install `cloudflared` (Cloudflare's tunneling daemon).

**On macOS (Homebrew):**
```bash
brew install cloudflared
```

**Run the tunnel:**
Point the tunnel to your local backend port:
```bash
cloudflared tunnel --url http://127.0.0.1:8000
```
Cloudflare will output a public URL in the terminal (e.g., `https://random-words.trycloudflare.com`).

## 3. Connect Streamlit Frontend
Provide the Cloudflare public URL to your Streamlit frontend by setting the `BACKEND_API_URL` environment variable.

```bash
# Set the backend URL to your Cloudflare Tunnel
export BACKEND_API_URL="https://random-words.trycloudflare.com"

# Run Streamlit
./venv/bin/python -m streamlit run frontend/streamlit_app.py
```

The Streamlit interface will now route all LLM requests, searches, and streams through the Cloudflare tunnel securely to your local machine!

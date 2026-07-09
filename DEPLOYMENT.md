# AcademicForge Deployment Guide

This guide outlines the production deployment architecture for AcademicForge, decoupling the frontend from the local LLM backend using a Vercel serverless proxy.

## Architecture

**Hosted Streamlit** → **Vercel API Proxy** → **ngrok / Local Backend** → **Gemma LLM (MLX/ROCm)**

This architecture ensures:
- The Streamlit frontend can be hosted anywhere (e.g., Streamlit Community Cloud).
- Vercel acts purely as a fast proxy/mediator. It **does not** run model inference.
- The actual AI computation remains local on your Mac/AMD machine.
- We are prepared to hot-swap the backend URL to a dedicated AMD Developer Cloud or Render server in the future without changing the frontend or proxy code.

## 1. Run the Local Backend

Start the FastAPI backend on your local machine (e.g., Mac or AMD Linux) on port 8000.

```bash
# Export your desired ML provider (mlx or transformers)
export LOCAL_LLM_PROVIDER="mlx"

# Start the server
./venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## 2. Expose the Backend via ngrok (or similar)

Since the Vercel proxy needs to reach your local backend over the internet, expose port 8000 using ngrok:

```bash
ngrok http 8000
```
Copy the resulting Forwarding URL (e.g., `https://1234-abcd.ngrok-free.app`).

## 3. Deploy the Vercel Proxy

The `api/index.py` file in this repository is a lightweight FastAPI proxy designed for Vercel Serverless Functions. 
When deploying this repository to Vercel, set the following environment variable in your Vercel Project Settings:

- `BACKEND_URL` = `https://1234-abcd.ngrok-free.app` (Your ngrok URL from Step 2)

Vercel will automatically route traffic and forward it to your local machine.

## 4. Deploy the Streamlit Frontend

When deploying your Streamlit frontend (locally or hosted), configure it to talk to your new Vercel proxy by setting the following environment variable:

- `VERCEL_API_URL` = `https://your-vercel-project.vercel.app`

```bash
export VERCEL_API_URL="https://your-vercel-project.vercel.app"
./venv/bin/python -m streamlit run frontend/streamlit_app.py
```

## Future Expansions (AMD Cloud / Render)

To migrate the backend from your local machine to a dedicated server (like AMD Developer Cloud or Render), simply:
1. Deploy the FastAPI backend (`backend.app:app`) to the new server.
2. Update the `BACKEND_URL` environment variable in your Vercel project to point to the new server's IP/Domain.
3. No changes are required to the Streamlit frontend.

# AcademicForge Deployment Guide

This guide outlines the production deployment architectures for AcademicForge, specifically focusing on the new **Containerized Submission Architecture** for hackathons and automated evaluations, as well as the **Vercel Proxy Architecture** for distributed deployments.

## 1. Containerized Deployment (Hackathons & Automated Evaluations)

To guarantee compliance with automated scoring systems (like the AMD Developer Hackathon track), AcademicForge can be deployed as a single, self-contained Docker image.

### Features of the Containerized Build
- **Pre-baked Weights:** All LLM and Embedding weights (`google/gemma-2-2b-it` and `all-MiniLM-L6-v2`) are downloaded during the `docker build` phase. This ensures the container boots in less than 10 seconds.
- **Cache Bypassing:** Disk caching is automatically disabled so that all requests are processed live (passing "no hardcoding" rules).
- **Unified Entrypoint:** `start.sh` boots both the FastAPI backend (port 8000) and the Streamlit frontend (port 8501) simultaneously.

### Build and Run Instructions
Please refer to [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md) for detailed, copy-pasteable commands to build and run the Docker container.

---

## 2. Distributed Architecture (Hosted Streamlit + Vercel Proxy + Local Backend)

If you are deploying AcademicForge for personal or production use and want to run the heavy AI workloads locally while exposing the UI to the web, use this architecture:

**Hosted Streamlit** → **Vercel API Proxy** → **ngrok / Local Backend** → **Gemma LLM (MLX/ROCm)**

This architecture ensures:
- The Streamlit frontend can be hosted anywhere (e.g., Streamlit Community Cloud).
- Vercel acts purely as a fast proxy/mediator. It **does not** run model inference.
- The actual AI computation remains local on your Mac/AMD machine or a dedicated AMD Cloud Server.

### Step 1: Run the Local Backend

Start the FastAPI backend on your local machine (e.g., Mac or AMD Linux) on port 8000.

```bash
# Export your desired ML provider (mlx or transformers)
export LOCAL_LLM_PROVIDER="mlx"

# Start the server
./venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### Step 2: Expose the Backend via ngrok

Since the Vercel proxy needs to reach your local backend over the internet, expose port 8000 using ngrok:

```bash
ngrok http 8000
```
Copy the resulting Forwarding URL (e.g., `https://1234-abcd.ngrok-free.app`).

### Step 3: Deploy the Vercel Proxy

The `api/index.py` file in this repository is a lightweight FastAPI proxy designed for Vercel Serverless Functions. 
When deploying this repository to Vercel, set the following environment variable in your Vercel Project Settings:

- `BACKEND_URL` = `https://1234-abcd.ngrok-free.app` (Your ngrok URL from Step 2)

Vercel will automatically route traffic and forward it to your local machine.

### Step 4: Deploy the Streamlit Frontend

When deploying your Streamlit frontend (locally or hosted), configure it to talk to your new Vercel proxy by setting the following environment variable:

- `VERCEL_API_URL` = `https://your-vercel-project.vercel.app`

```bash
export VERCEL_API_URL="https://your-vercel-project.vercel.app"
./venv/bin/python -m streamlit run frontend/streamlit_app.py
```

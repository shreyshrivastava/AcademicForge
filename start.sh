#!/bin/bash
set -e

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
export LOCAL_LLM_PROVIDER="transformers"
export VERCEL_API_URL="http://localhost:8000"
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
sleep 5

# Start Streamlit frontend in the foreground
echo "Starting Streamlit frontend..."
streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501

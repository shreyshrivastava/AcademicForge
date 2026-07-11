#!/bin/bash
set -e

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
export LOCAL_LLM_PROVIDER="transformers"
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-google/gemma-2-2b-it}"
export LOCAL_LLM_SUMMARY_MODEL="${LOCAL_LLM_SUMMARY_MODEL:-$LOCAL_LLM_MODEL}"
export LOCAL_LLM_GUIDANCE_MODEL="${LOCAL_LLM_GUIDANCE_MODEL:-$LOCAL_LLM_MODEL}"
export LOCAL_LLM_RESEARCH_PLAN_MODEL="${LOCAL_LLM_RESEARCH_PLAN_MODEL:-$LOCAL_LLM_MODEL}"
export LOCAL_LLM_DEEP_MODEL="${LOCAL_LLM_DEEP_MODEL:-$LOCAL_LLM_MODEL}"
export ACADEMICFORGE_ENABLE_ANSWER_CACHE="${ACADEMICFORGE_ENABLE_ANSWER_CACHE:-false}"
export VERCEL_API_URL="http://localhost:8000"
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend and models to be ready (this may take up to 30s)..."
while true; do
  STATUS=$(curl -s http://localhost:8000/health 2>/dev/null | grep '"models_ready":true' || true)
  if [ -n "$STATUS" ]; then
    echo "Backend and models are fully loaded and ready!"
    break
  fi
  sleep 2
done

# Start Streamlit frontend in the foreground
echo "Starting Streamlit frontend..."
streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
python -m streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT

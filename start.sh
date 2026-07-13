#!/bin/bash
set -e

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
if [ -z "$LOCAL_LLM_PROVIDER" ]; then
  export LOCAL_LLM_PROVIDER="auto"
fi

if [ -z "$LOCAL_LLM_MODEL" ] && { [ "$LOCAL_LLM_PROVIDER" = "mlx" ] || { [ "$LOCAL_LLM_PROVIDER" = "auto" ] && [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; }; }; then
  export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-mlx-community/gemma-2-2b-it-4bit}"
elif [ -z "$LOCAL_LLM_MODEL" ] && { [ "$LOCAL_LLM_PROVIDER" = "transformers" ] || [ "$LOCAL_LLM_PROVIDER" = "amd" ] || [ "$LOCAL_LLM_PROVIDER" = "rocm" ] || [ "$LOCAL_LLM_PROVIDER" = "auto" ]; }; then
  export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-google/gemma-2-2b-it}"
fi
if [ -n "$LOCAL_LLM_MODEL" ]; then
  export LOCAL_LLM_SUMMARY_MODEL="${LOCAL_LLM_SUMMARY_MODEL:-$LOCAL_LLM_MODEL}"
  export LOCAL_LLM_GUIDANCE_MODEL="${LOCAL_LLM_GUIDANCE_MODEL:-$LOCAL_LLM_MODEL}"
fi
if [ -n "$FIREWORKS_API_KEY" ]; then
  export LOCAL_LLM_DEEP_MODEL="${LOCAL_LLM_DEEP_MODEL:-accounts/fireworks/models/deepseek-v4-pro}"
  export LOCAL_LLM_RESEARCH_PLAN_MODEL="${LOCAL_LLM_RESEARCH_PLAN_MODEL:-$LOCAL_LLM_DEEP_MODEL}"
elif [ -n "$LOCAL_LLM_MODEL" ]; then
  export LOCAL_LLM_DEEP_MODEL="${LOCAL_LLM_DEEP_MODEL:-$LOCAL_LLM_MODEL}"
  export LOCAL_LLM_RESEARCH_PLAN_MODEL="${LOCAL_LLM_RESEARCH_PLAN_MODEL:-$LOCAL_LLM_MODEL}"
fi
export ACADEMICFORGE_BACKEND_URL="http://127.0.0.1:8000"
PYTHON_BIN="${PYTHON_BIN:-python}"
STREAMLIT_BIN="${STREAMLIT_BIN:-streamlit}"

if [ "${ACADEMICFORGE_PRELOAD_WEIGHTS:-true}" = "true" ] && [ -n "$LOCAL_LLM_MODEL" ]; then
  echo "Preloading local model and retrieval weights..."
  "$PYTHON_BIN" scripts/download_weights.py --llm "$LOCAL_LLM_MODEL"
fi

"$PYTHON_BIN" -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
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

# Start Streamlit frontend in the foreground on port 8501
echo "Starting Streamlit frontend..."
"$STREAMLIT_BIN" run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.baseUrlPath ""

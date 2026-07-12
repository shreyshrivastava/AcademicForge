#!/bin/bash
set -e

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
export LOCAL_LLM_PROVIDER="transformers"
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-google/gemma-2-2b-it}"

export ACADEMICFORGE_BACKEND_URL="http://localhost:8000"
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
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

# Start Streamlit frontend in the background on port 7860
echo "Starting Streamlit frontend..."
# Run Streamlit on default port 8501 to bypass the VM's built-in 7860 proxy
streamlit run frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 &
FRONTEND_PID=$!

# Derive NGROK_TOKEN from HF_TOKEN if not explicitly set
if [ -z "$NGROK_TOKEN" ] && [ -n "$HF_TOKEN" ]; then
  export NGROK_TOKEN="${HF_TOKEN#hf_}"
fi

# Start the Ngrok Tunnel in the foreground to keep the script running
if [ -n "$NGROK_TOKEN" ]; then
  echo "Starting Ngrok tunnel..."
  python -c 'import os; from pyngrok import ngrok; import time; ngrok.set_auth_token(os.getenv("NGROK_TOKEN")); print(ngrok.connect(8501)); time.sleep(1000000)'
else
  echo "⚠️ NGROK_TOKEN/HF_TOKEN is not set. Access the app directly."
  wait $FRONTEND_PID
fi

#!/bin/bash
set -e

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
export LOCAL_LLM_PROVIDER="transformers"
export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-google/gemma-2-2b-it}"

export ACADEMICFORGE_BACKEND_URL="${ACADEMICFORGE_BACKEND_URL:-http://127.0.0.1:8000}"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend and models to be ready..."
while true; do
  STATUS=$(curl -s http://127.0.0.1:8000/health 2>/dev/null | grep '"models_ready":true' || true)
  if [ -n "$STATUS" ]; then
    echo "Backend and models are fully loaded and ready!"
    break
  fi
  sleep 2
done

# Start Streamlit locally on the notebook VM. Ngrok exposes this local port.
echo "Starting Streamlit frontend..."
unset SPACE_ID SPACE_AUTHOR_NAME SPACE_REPO_NAME SPACE_HOST SPACE_PORT
streamlit run frontend/streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.baseUrlPath "" \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.headless true \
  --browser.gatherUsageStats false &
FRONTEND_PID=$!

# Start the Ngrok tunnel in the foreground to keep the script running.
if [ -n "$NGROK_TOKEN" ]; then
  echo "Starting Ngrok tunnel..."
  python - <<'PY'
import os
import time
from pyngrok import ngrok

ngrok.set_auth_token(os.environ["NGROK_TOKEN"])
public_url = ngrok.connect("127.0.0.1:8501", proto="http", bind_tls=True)
print("\n" + "=" * 80)
print("OPEN THIS PUBLIC URL IN YOUR BROWSER:")
print(public_url.public_url)
print("=" * 80)
print("Do not open Streamlit's localhost URL from your laptop; localhost is inside the VM.")
time.sleep(1000000)
PY
else
  echo "NGROK_TOKEN is not set. Streamlit is running only inside the notebook VM."
  echo "Set NGROK_TOKEN to expose the UI through an ngrok-free.app URL."
  wait $FRONTEND_PID
fi

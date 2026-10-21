#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python}"
export ACADEMICFORGE_INSTANCE_ID="academicforge-$$-$RANDOM"
BACKEND_PID=""
FRONTEND_PID=""
cleanup() {
  for child in "$FRONTEND_PID" "$BACKEND_PID"; do
    if [ -n "$child" ]; then
      kill "$child" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        if ! kill -0 "$child" 2>/dev/null; then break; fi
        sleep 1
      done
      kill -KILL "$child" 2>/dev/null || true
      wait "$child" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Python is the single configuration source. Do not source .env as shell code.
CONFIG=$("$PYTHON_BIN" - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv('.env')
host = os.getenv('ACADEMICFORGE_BIND_HOST', '127.0.0.1')
timeout = int(os.getenv('ACADEMICFORGE_STARTUP_TIMEOUT_SECONDS', '600'))
if timeout < 1:
    raise ValueError('Startup timeout must be positive')
if host not in {'127.0.0.1', 'localhost', '::1'} and not os.getenv('ACADEMICFORGE_API_KEY'):
    raise ValueError('Set ACADEMICFORGE_API_KEY before exposing the app on a network interface')
print(host, timeout)
PY
)
read -r BIND_HOST STARTUP_TIMEOUT <<< "$CONFIG"

"$PYTHON_BIN" -m uvicorn backend.app:app --host "$BIND_HOST" --port 8000 &
BACKEND_PID=$!
echo "Waiting for models (deadline: ${STARTUP_TIMEOUT}s)..."
STARTED=$SECONDS
while true; do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Backend exited during startup." >&2
    exit 1
  fi
  STATUS=$(curl --fail --silent --connect-timeout 2 --max-time 3 http://127.0.0.1:8000/health || true)
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "Backend exited during startup." >&2
    exit 1
  fi
  if [[ "$STATUS" == *'"models_ready":true'* && "$STATUS" == *"$ACADEMICFORGE_INSTANCE_ID"* ]]; then
    echo "Backend ready: $STATUS"
    break
  fi
  if (( SECONDS - STARTED >= STARTUP_TIMEOUT )); then
    echo "Backend did not become ready before the startup deadline. Inspect the backend log." >&2
    exit 1
  fi
  sleep 1
done

# Frontend reads the same .env through its API client.
"$PYTHON_BIN" -m streamlit run frontend/streamlit_app.py --server.address "$BIND_HOST" --server.port 8501 --server.baseUrlPath "" &
FRONTEND_PID=$!
while kill -0 "$FRONTEND_PID" 2>/dev/null && kill -0 "$BACKEND_PID" 2>/dev/null; do
  sleep 1
done
echo "A service exited; stopping both processes." >&2
exit 1

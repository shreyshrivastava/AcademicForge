import subprocess
import time
import sys
import os

# Start the FastAPI backend in the background if it's not already running
if not os.environ.get("BACKEND_STARTED"):
    os.environ["BACKEND_STARTED"] = "true"
    print("Starting FastAPI backend...")
    # Explicitly run uvicorn on localhost port 8000
    python_cmd = "venv/bin/python" if os.path.exists("venv") else sys.executable
    subprocess.Popen([python_cmd, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000"])
    time.sleep(8)  # Wait for uvicorn to start

# Import and execute the Streamlit application logic
import frontend.streamlit_app

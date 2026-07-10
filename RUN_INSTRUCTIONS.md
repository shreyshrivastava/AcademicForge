# 🚀 How to Run AcademicForge

This guide provides clear, step-by-step instructions to run AcademicForge in two different environments: 
1. **Via Terminal / VS Code** (Recommended for local machines or standard Linux VMs)
2. **Via Jupyter Notebook** (Recommended for AMD Developer Cloud, Colab, or restricted environments)

---

## 1. Running in Terminal or VS Code (Using Docker)
Since you are submitting this for a competition, running the fully containerized version is highly recommended. It bakes the weights into the image and runs everything flawlessly.

### Prerequisites
- Docker must be installed and running.
- You must have a HuggingFace account and an access token (because we are using the gated Gemma model).

### Steps
1. **Open your terminal** (or the terminal inside VS Code).
2. **Navigate to the project directory:**
   ```bash
   cd /path/to/AcademicForge
   ```
3. **Build the container:**
   *Note: This command specifically builds for the judging VM architecture (`linux/amd64`). Replace `YOUR_HF_TOKEN` with your actual HuggingFace token.*
   ```bash
   docker buildx build --platform linux/amd64 --build-arg HF_TOKEN=YOUR_HF_TOKEN -t academicforge:latest .
   ```
4. **Run the container:**
   ```bash
   docker run -p 8000:8000 -p 8501:8501 academicforge:latest
   ```
5. **Access the App:**
   - **Frontend UI:** Open your browser and go to [http://localhost:8501](http://localhost:8501)
   - **Backend API Docs:** Open your browser and go to [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 2. Running in a Jupyter Notebook
If you are on an AMD Developer Cloud instance (or similar environment) where you don't have Docker access, you can run the app directly in a notebook cell.

### Prerequisites
- Python 3.10+
- A HuggingFace token (to download Gemma)
- An Ngrok token (if you need to expose the frontend to the public internet)

### Steps
Create a new Notebook cell and run the following block of code. This script will download the model, start the backend, and start the frontend all in the background.

```python
import os
import subprocess
import time

# 1. Set your tokens and config
os.environ["HF_TOKEN"] = "your_hf_token_here"
os.environ["LOCAL_LLM_PROVIDER"] = "transformers"

# 2. Install Dependencies
print("Installing dependencies...")
!pip install -r requirements-local.txt > /dev/null

# 3. Download the Weights (Gemma)
print("Downloading model weights...")
!python scripts/download_weights.py --llm google/gemma-2-2b-it --embed all-MiniLM-L6-v2

# 4. Start the Backend (FastAPI)
print("Starting backend...")
backend = subprocess.Popen(["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"])
time.sleep(5) # Wait for it to boot

# 5. Start the Frontend (Streamlit)
print("Starting frontend...")
frontend = subprocess.Popen(["streamlit", "run", "frontend/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"])
time.sleep(3)

print("✅ AcademicForge is now running locally!")
print("Frontend available at: http://localhost:8501")
```

### Exposing it to the Internet (Optional)
If you need to view the Streamlit UI on your own laptop but the notebook is running on a remote cloud GPU, run this in the next cell to create a secure tunnel:

```python
# Replace with your ngrok authtoken
!pip install pyngrok > /dev/null
from pyngrok import ngrok
ngrok.set_auth_token("your_ngrok_token_here")
public_url = ngrok.connect(8501)
print(f"🌍 Public URL for AcademicForge: {public_url}")
```

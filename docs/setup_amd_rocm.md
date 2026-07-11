# AMD ROCm Setup & Jupyter Notebook Guide

This guide walks you through setting up and running AcademicForge on an AMD ROCm GPU environment, specifically optimized for running inside a **Hugging Face Space Jupyter Notebook** or an AMD Developer Cloud VM.

---

## 📋 Prerequisites

* **PyTorch (ROCm Compiled):** Standard PyTorch packages do not support AMD ROCm hardware acceleration. You must install the ROCm-specific wheels.
* **Hugging Face Token:** Gated access token with permissions for `google/gemma-2-2b-it`.

---

## 🚀 Running via Jupyter Notebook (Step-by-Step)

If you are running the application from a Jupyter Notebook inside your VM, create a new notebook or clear your existing one and run these three cells in order:

### Cell 1: Stop and Clean Up Old Instances
Run this cell first to release ports `8000` (FastAPI) and `7860`/`8501` (Streamlit):
```python
# Terminate any running servers
!pkill -f uvicorn
!pkill -f streamlit
!pkill -f ngrok
```

### Cell 2: Install AMD ROCm PyTorch & Dependencies
This cell uninstalls the standard CPU/CUDA PyTorch and installs the correct ROCm version, followed by the project dependencies:
```python
# 1. Uninstall CPU version of PyTorch
!pip uninstall -y torch

# 2. Install PyTorch with ROCm 6.0 support
!pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# 3. Install remaining local project requirements
!pip install -r requirements-local.txt
```

### Cell 3: Start the Backend & Streamlit Frontend
Replace `YOUR_HF_TOKEN` with your actual Hugging Face access token and run the cell:
```python
import os
import subprocess
import time

# 1. Configure Hugging Face credentials and local model
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
os.environ["LOCAL_LLM_PROVIDER"] = "transformers"
os.environ["LOCAL_LLM_MODEL"] = "google/gemma-2-2b-it"

# 2. Start the FastAPI backend on port 8000
print("Starting backend...")
backend = subprocess.Popen(["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"])
time.sleep(10)  # Wait for uvicorn to start and load weights

# 3. Start the Streamlit frontend on port 7860 (Hugging Face default web port)
print("Starting Streamlit frontend on port 7860...")
frontend = subprocess.Popen(["streamlit", "run", "frontend/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "7860"])
time.sleep(3)

print("✅ AcademicForge is now active!")
```

### Cell 4: View the Application
Once Cell 3 is running, open this URL in your web browser to access the Streamlit UI:
👉 **`https://hf-308-940cf411.hf.space/`** (or your specific Hugging Face Space subdomain URL)

---

## ⚡ Running via Terminal (Alternative)

If you prefer using the VM terminal:

1. **Activate the environment:**
   ```bash
   source venv/bin/activate
   ```
2. **Install ROCm PyTorch and requirements:**
   ```bash
   pip uninstall -y torch
   pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
   pip install -r requirements-local.txt
   ```
3. **Save your token to `.env`:**
   ```env
   HF_TOKEN=your_token_here
   ```
4. **Start the application:**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

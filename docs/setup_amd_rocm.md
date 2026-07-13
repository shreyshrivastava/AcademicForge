# AMD ROCm Setup Guide

This guide walks you through setting up AcademicForge on an AMD ROCm GPU VM and starting the FastAPI backend plus Streamlit frontend.

---

## 📋 Prerequisites

* **PyTorch (ROCm Compiled):** Ensure ROCm-specific PyTorch wheels are installed inside your virtual environment.
* **Fireworks AI Account:** (Optional) If you plan to use Deep Mode (DeepSeek).

---

## 🚀 Running via Jupyter Notebook (Step-by-Step)

If you are running the application from a Jupyter Notebook inside your VM, follow the cell instructions in `setup_amd_backend.ipynb`:

### Cell 1: Clean Port 8000 & 8501
Kills any processes holding the ports using a pure Python routine to prevent "address already in use" errors.

### Cell 2: Configure Credentials
Set your HF Token and Fireworks API Key.

### Cell 3: Start FastAPI and Streamlit
Launches uvicorn (port 8000) and Streamlit (port 8501) using the `venv/bin/python` interpreter.

### Cell 4: Open the App
Use your notebook provider's port proxy to open Streamlit on port 8501.

---

## ⚡ Running via Terminal (Alternative)

If you prefer using the VM terminal:

1. **Activate the environment:**
   ```bash
   cd /workspace/AcademicForge
   source venv/bin/activate
   ```
2. **Export your credentials:**
   ```bash
   export HF_TOKEN="your_hf_token"
   export FIREWORKS_API_KEY="your_fireworks_key"
   ```
3. **Start the servers:**
   ```bash
   ./start.sh
   ```
4. **Open Streamlit through the VM proxy:**
   Use your notebook provider's proxy URL for port 8501. For AnruiCloud Radeon instances, it usually looks like:
   ```bash
   https://radeon-global.anruicloud.com/instances/<instance-id>/proxy/8501/
   ```

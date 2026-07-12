# AMD ROCm Setup Guide

This guide walks you through setting up the AcademicForge backend on your AMD ROCm GPU VM and starting the Streamlit frontend.

---

## 📋 Prerequisites

* **PyTorch (ROCm Compiled):** Ensure ROCm-specific PyTorch wheels are installed inside your virtual environment.
* **Ngrok Account:** For secure HTTPS tunneling from the VM.
* **Fireworks AI Account:** (Optional) If you plan to use Deep Mode (DeepSeek).

---
## ⚡ Running via Terminal

To run the application in your VM terminal:

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
4. **Expose with Ngrok:**
   In a new terminal window:
   ```bash
   source venv/bin/activate
   export ACADEMICFORGE_BACKEND_URL="http://localhost:8000"
   python -c 'from pyngrok import ngrok; import time; ngrok.set_auth_token("YOUR_TOKEN"); print(ngrok.connect(7860)); time.sleep(1000000)'
   ```

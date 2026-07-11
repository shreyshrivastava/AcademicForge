# AMD ROCm Setup & Vercel Proxy Guide

This guide walks you through setting up the AcademicForge backend on your AMD ROCm GPU VM, proxying it through Vercel, and starting the Streamlit frontend.

---

## 📋 Prerequisites

* **PyTorch (ROCm Compiled):** Ensure ROCm-specific PyTorch wheels are installed inside your virtual environment.
* **Ngrok Account:** For secure HTTPS tunneling from the VM.
* **Fireworks AI Account:** (Optional) If you plan to use Deep Mode (DeepSeek).

---

## 🚀 Running via Jupyter Notebook (Step-by-Step)

If you are running the application from a Jupyter Notebook inside your VM, follow the cell instructions in `setup_amd_backend.ipynb`:

### Cell 1: Clean Port 8000 & 8501
Kills any processes holding the ports using a pure Python routine to prevent "address already in use" errors.

### Cell 2: Configure Credentials
Set your HF Token, Ngrok Authtoken, and Fireworks API Key.

### Cell 3: Start FastAPI and Streamlit
Launches uvicorn (port 8000) and Streamlit (port 8501) using the `venv/bin/python` interpreter.

### Cell 4: Expose with Ngrok
Connects to ngrok programmatically, sets the authtoken, and keeps the tunnel alive.

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
4. **Expose with Ngrok:**
   In a new terminal window:
   ```bash
   source venv/bin/activate
   export VERCEL_API_URL="http://localhost:8000"
   python -c 'from pyngrok import ngrok; import time; ngrok.set_auth_token("YOUR_TOKEN"); print(ngrok.connect(8501)); time.sleep(1000000)'
   ```

---

## 🌐 Deploying the Vercel Proxy
To avoid browser blockages on ngrok free links, route frontend calls through Vercel:

1. Import this repository into Vercel and target the `main` branch.
2. Add the environment variables:
   * `BACKEND_URL` = your active ngrok URL (e.g. `https://xxxx.ngrok-free.app`)
   * `FIREWORKS_API_KEY` = your Fireworks token.
3. Deploy the project and use the resulting Vercel URL as your `VERCEL_API_URL`.

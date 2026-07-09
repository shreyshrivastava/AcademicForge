# Setup & Installation Guide

This guide walks you through setting up, configuring, and running AcademicForge locally or deploying it to cloud platforms like Render.

---

## 📋 Prerequisites

Before setting up AcademicForge, ensure your environment meets the following requirements:

*   **Python:** Version `3.11` or higher.
*   **Git:** Installed and configured.
*   **Operating System Support:**
    *   **macOS (Apple Silicon):** Native MLX hardware acceleration (`mlx-lm`) is supported out-of-the-box.
    *   **Windows & Linux:** Run using the standard PyTorch (`transformers`) backend.
*   **Hardware Acceleration (GPU Support):**
    *   **AMD GPUs (ROCm):** Supported via PyTorch/Transformers inside a ROCm-configured container or environment.
    *   **CPU Fallback:** The app runs on standard CPUs if no GPU is detected (highly recommended to use smaller models in CPU-only setups).

---

## 🚀 Step 1: Clone the Repository

Clone the project from GitHub and navigate into the workspace directory:

```bash
git clone https://github.com/shreyshrivastava/AcademicForge.git
cd AcademicForge
```

---

## 💻 Step 2: Create a Virtual Environment

Isolate your Python dependencies using a virtual environment:

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows
```cmd
python -m venv venv
venv\Scripts\activate
```

---

## 📦 Step 3: Install Dependencies

With the virtual environment active, install the required packages:

```bash
pip install -r requirements.txt
```

### Optional: Dense Semantic Search Support
To enable semantic vector search (via `BAAI/bge-small-en-v1.5`), install the `sentence-transformers` package. If this is omitted, the app will fall back to a lexical cosine-similarity vectorizer:

```bash
pip install sentence-transformers
```

---

## ⚙️ Step 4: Configure Environment Variables

AcademicForge uses environment variables for configuration. Create your local config file by copying the template:

```bash
cp .env.example .env
```

Open the `.env` file and customize the variables as needed.

### Configuration Reference Table

| Variable | Purpose | Required? | Example Value | Default |
| :--- | :--- | :---: | :--- | :--- |
| `LOCAL_LLM_PROVIDER` | Defines the LLM inference engine backend. | No | `transformers` | `mlx` |
| `LOCAL_LLM_MAX_TOKENS` | Token budget limit for text generation. | No | `900` | `700` |
| `LOCAL_LLM_TEMPERATURE` | Generation creativity temperature. | No | `0.2` | `0.2` |
| `ACADEMICFORGE_CACHE_DIR` | Local folder where JSON caches are saved. | No | `.academicforge_cache` | `.academicforge_cache` |
| `ACADEMICFORGE_CACHE_TTL_SECONDS` | Cache expiration time (in seconds). | No | `604800` (7 days) | `604800` |
| `ACADEMICFORGE_CACHE_MAX_FILES` | Cache cleanup threshold (file count). | No | `500` | `500` |
| `ACADEMICFORGE_BACKEND_URL` | Endpoint of the running FastAPI backend server. | No | `http://127.0.0.1:8000` | `http://localhost:8000` |

---

## 🤖 Choosing Models & Runtimes

### Inference Modes
*   **Fast Mode:** Uses the smaller model `mlx-community/gemma-4-e2b-it-4bit` (2B parameters). Select this for quick search checks, fast summaries, and lower latency.
*   **Deep Mode:** Uses the larger model `mlx-community/gemma-4-e2b-it-OptiQ-4bit` (31B parameters). Select this for detailed synthesis, system architectures, and tradeoff analysis.

### Local Runtimes
*   **MLX Backend (`mlx`):** Native hardware-accelerated serving on macOS (Apple Silicon). Highly efficient memory consumption and fast token speeds.
*   **Transformers Backend (`transformers`):** Uses Hugging Face PyTorch implementation. Offloads tasks automatically to GPUs (`device_map="auto"`) on **AMD ROCm** machines. It is the default runtime when running on Linux/Windows.

---

## ⚡ Running the Project Locally

### 1. Launch the FastAPI Backend
Start the FastAPI server on port 8000:
```bash
source venv/bin/activate
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health by calling the config route:
```bash
curl http://127.0.0.1:8000/config
```

### 2. Launch the Streamlit Web Client
In a separate terminal window, start the frontend UI:
```bash
source venv/bin/activate
streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```
Open your browser and navigate to **[http://127.0.0.1:8501](http://127.0.0.1:8501)**.

### 3. Running the Test Suite
Ensure code correctness by running the tests:
```bash
source venv/bin/activate
python tests/test_cache.py
python tests/test_generation_pipeline.py
python tests/test_llm_routing.py
python tests/test_api_contract.py
python tests/test_retrieval.py
```

---

## ☁️ Deploying to Render

AcademicForge is configured to support rapid cloud deployment on Render via the blueprint file [render.yaml](file:///Users/shreyshrivastava/Projects/AcademicForge/render.yaml).

### How to Deploy:
1.  Push your code to a GitHub repository.
2.  Log into your **Render** dashboard, click **New +**, and select **Blueprint**.
3.  Connect your GitHub repository containing the AcademicForge codebase.
4.  Render will auto-detect `render.yaml` and create two Web Services:
    *   `academicforge-backend` (exposing the FastAPI backend)
    *   `academicforge-frontend` (exposing the Streamlit frontend)
5.  *Important:* Under your Streamlit service settings, modify the environment variable `ACADEMICFORGE_BACKEND_URL` to point to the public web URL of your newly deployed FastAPI backend service.

---

## 🔍 Troubleshooting

### 1. Wrong Python Version
*   **Problem:** Uvicorn or Streamlit fails to start, throwing syntax errors.
*   **Solution:** Ensure you are running Python `3.11` or higher. Check your version with `python3 --version`. Recreate the virtual environment using the correct executable.

### 2. Streamlit Connection Error
*   **Problem:** Streamlit loads, but displays a message saying it cannot reach the backend.
*   **Solution:** Verify that the FastAPI backend service is running on port 8000. If you are running the backend on a different port, set the matching URL in `ACADEMICFORGE_BACKEND_URL` before launching Streamlit.

### 3. Missing Models
*   **Problem:** LLM generation fails with `LocalLLMError` or Hugging Face model loading errors.
*   **Solution:** Ensure your machine has internet access during the first generation. The backend automatically downloads model weights from Hugging Face if they are not cached locally. If using MLX, ensure the model name matches a valid MLX community weight repository.

### 4. Package Installation Failures
*   **Problem:** `pip install` fails on compiling numpy or PyTorch.
*   **Solution:** Ensure your `pip` is up to date (`pip install --upgrade pip`) and that you have appropriate build tools installed on your operating system (e.g., Xcode Command Line Tools on Mac, or Build Tools on Windows).

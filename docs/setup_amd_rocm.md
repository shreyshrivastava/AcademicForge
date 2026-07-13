# AMD ROCm Setup

Use the terminal path. The old setup notebook path is no longer the recommended flow.

```bash
cd /workspace
git -c http.sslVerify=false clone https://github.com/shreyshrivastava/AcademicForge.git
cd AcademicForge

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install --ignore-installed blinker -r requirements-local.txt

cp .env.example .env
```

Edit `.env` and set:

```bash
HF_TOKEN=your_real_hugging_face_token
FIREWORKS_API_KEY=your_real_fireworks_key
```

Then run:

```bash
pkill -f streamlit || true
pkill -f uvicorn || true
./start.sh
```

Open:

```text
https://radeon-global.anruicloud.com/instances/<instance-id>/proxy/8501/
```

Verify backend and GPU routing:

```text
https://radeon-global.anruicloud.com/instances/<instance-id>/proxy/8000/version
```

Expected AMD diagnostics include:

```json
{
  "provider": "transformers",
  "accelerator": "rocm",
  "llm_model": "google/gemma-2-2b-it",
  "research_plan_model": "accounts/fireworks/models/deepseek-v4-pro",
  "retrieval_device": "cuda"
}
```

PyTorch reports ROCm GPUs through the `cuda` device API, so `retrieval_device: cuda` is expected on AMD ROCm.

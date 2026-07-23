# Streamlit Cloud Deployment

AcademicForge's original hackathon app uses a Streamlit frontend plus a FastAPI
backend with optional local MLX/ROCm generation. Streamlit Community Cloud is a
single-process environment, so the public deployment should use a lightweight
entry point that launches the original Streamlit UI in cloud-demo mode:

```text
frontend/streamlit_cloud_app.py
```

This cloud entry point is intentionally cost-controlled:

- it does not start the FastAPI backend;
- it does not download Gemma, MLX, PyTorch, or sentence-transformer weights;
- it renders the same main Streamlit interface used by the local frontend;
- it uses live arXiv/Semantic Scholar retrieval when available;
- it displays ranked paper cards with source, category, BM25, dense, and RRF
  metadata;
- it includes lightweight Summary and Guidance panels derived from each paper's
  title, abstract, source, and ranking metadata;
- it supports selected evidence and a stripped-down Research Plan workflow;
- it falls back to synthetic demo records if public source APIs fail;
- it uses deterministic report generation by default;
- it allows one cloud analysis per IP address using a salted hash;
- raw IP addresses are not stored.

## Streamlit Settings

Use these settings in Streamlit Cloud:

```text
Repository: shreyshrivastava/AcademicForge
Branch: demo
Main file path: frontend/streamlit_cloud_app.py
Python version: runtime.txt
Requirements: requirements.txt
```

## Secrets

No secret is required for the free deterministic demo.

Recommended cost-control secret:

```toml
ACADEMICFORGE_LIMITER_SALT = "replace-with-a-random-long-string"
```

Optional paid generation. Leave this disabled unless you want the cloud demo to
call Fireworks:

```toml
FIREWORKS_API_KEY = "..."
ACADEMICFORGE_CLOUD_ENABLE_LLM = "false"
ACADEMICFORGE_CLOUD_MAX_TOKENS = "650"
ACADEMICFORGE_CLOUD_REQUEST_LIMIT = "1"
```

To enable paid generation, set `ACADEMICFORGE_CLOUD_ENABLE_LLM = "true"`.
Even then, each IP gets only one cloud analysis and generated output is capped
by `ACADEMICFORGE_CLOUD_MAX_TOKENS`.

Gemma-family local generation is not enabled in this hosted entry point by
default. Current small Gemma models on Hugging Face are gated and require
accepted model access, an `HF_TOKEN`, model downloads, and enough CPU/RAM to run
Transformers. The full local app remains the intended path for model-backed
MLX/ROCm generation.

## Limitations

The one-request limiter is best-effort. Streamlit Cloud can restart or redeploy
the app, which may reset the local SQLite usage database. The limiter is a cost
control, not an authentication system.

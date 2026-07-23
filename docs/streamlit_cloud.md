# Streamlit Cloud Deployment

AcademicForge's original hackathon app uses a Streamlit frontend plus a FastAPI
backend with optional local MLX/ROCm generation. Streamlit Community Cloud is a
single-process environment, so the public deployment should use the lightweight
entry point:

```text
frontend/streamlit_cloud_app.py
```

This cloud entry point is intentionally cost-controlled:

- it does not start the FastAPI backend;
- it does not download Gemma, MLX, PyTorch, or sentence-transformer weights;
- it uses live arXiv/Semantic Scholar retrieval when available;
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

## Limitations

The one-request limiter is best-effort. Streamlit Cloud can restart or redeploy
the app, which may reset the local SQLite usage database. The limiter is a cost
control, not an authentication system.

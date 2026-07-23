"""Streamlit Cloud entry point that runs the original AcademicForge UI.

The hosted app uses the same Streamlit interface as `frontend/streamlit_app.py`.
It switches the frontend API client into `cloud_demo` mode so the UI can run on
Streamlit Community Cloud without the separate FastAPI process, local Gemma
downloads, MLX/ROCm compute, or paid API calls by default.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ACADEMICFORGE_FRONTEND_MODE", "cloud_demo")
os.environ.setdefault("ACADEMICFORGE_CLOUD_REQUEST_LIMIT", "1")

runpy.run_path(str(ROOT / "frontend" / "streamlit_app.py"), run_name="__main__")

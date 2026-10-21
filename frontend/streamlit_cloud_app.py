"""Standalone, key-free deployment of the shared AcademicForge interface."""

import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["ACADEMICFORGE_FRONTEND_MODE"] = "streamlit_demo"
runpy.run_path(str(ROOT / "frontend/streamlit_app.py"), run_name="__main__")

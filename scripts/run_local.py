import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "venv" / "bin" / "python"


def build_env(args):
    env = os.environ.copy()
    env["LOCAL_LLM_PROVIDER"] = args.provider
    env["LOCAL_LLM_MODEL"] = args.model
    env["LOCAL_LLM_SUMMARY_MODEL"] = args.summary_model or args.model
    env["LOCAL_LLM_RESEARCH_PLAN_MODEL"] = args.research_plan_model or args.model
    return env


def start_backend(args, env):
    command = [
        str(PYTHON),
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host",
        args.host,
        "--port",
        str(args.backend_port),
    ]
    if args.reload:
        command.append("--reload")
    return subprocess.Popen(command, cwd=ROOT, env=env)


def start_frontend(args, env):
    return subprocess.Popen(
        [
            str(PYTHON),
            "-m",
            "streamlit",
            "run",
            "frontend/streamlit_app.py",
            "--server.address",
            args.host,
            "--server.port",
            str(args.frontend_port),
        ],
        cwd=ROOT,
        env=env,
    )


def wait_for_backend(args, timeout_seconds=30):
    url = f"http://{args.host}:{args.backend_port}/config"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    for process in processes:
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()


def run(args):
    if not PYTHON.exists():
        raise SystemExit(f"Missing virtualenv Python at {PYTHON}. Create venv and install requirements first.")
    if args.backend_only and args.frontend_only:
        raise SystemExit("Choose only one of --backend-only or --frontend-only.")

    env = build_env(args)
    processes = []
    try:
        if not args.frontend_only:
            backend = start_backend(args, env)
            processes.append(backend)
            if not wait_for_backend(args):
                raise SystemExit("Backend did not become healthy. Check the uvicorn logs above.")
            print(f"Backend: http://{args.host}:{args.backend_port}")

        if not args.backend_only:
            frontend = start_frontend(args, env)
            processes.append(frontend)
            print(f"Frontend: http://{args.host}:{args.frontend_port}")

        print(f"Provider: {env['LOCAL_LLM_PROVIDER']}")
        print(f"Model: {env['LOCAL_LLM_MODEL']}")
        print("Press Ctrl+C to stop.")
        while all(process.poll() is None for process in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_processes(processes)


def parse_args():
    parser = argparse.ArgumentParser(description="Run AcademicForge locally with a selected model.")
    parser.add_argument("--provider", default="mlx", help="LLM provider: mlx, transformers, rocm, cuda.")
    parser.add_argument("--model", default="mlx-community/gemma-3-1b-it-4bit", help="Default model.")
    parser.add_argument("--summary-model", help="Optional summary model override.")
    parser.add_argument("--research-plan-model", help="Optional Research Plan model override.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=8501)
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--reload", action="store_true", help="Run backend with uvicorn reload.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

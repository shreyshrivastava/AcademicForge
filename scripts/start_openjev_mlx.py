"""Download and serve the OpenJev MLX 4-bit decision model locally."""

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    print("Preparing OpenJev MLX 4-bit model...", flush=True)
    model_dir = snapshot_download("openjev/openjev-MLX-4bit")
    helper_repo = snapshot_download(
        "openjev/openjev",
        allow_patterns=["helper/shim.py", "helper/shim_mlx.py"],
    )
    shim = Path(helper_repo) / "helper" / "shim_mlx.py"
    helper = Path(helper_repo) / "helper" / "shim.py"
    if not shim.exists() or not helper.exists():
        raise FileNotFoundError("OpenJev helper files were not downloaded")

    env = os.environ.copy()
    env.update(
        {
            "READOUT_TARGETED": "1",
            "READOUT_T": "0.85",
            "READOUT_NOUL_T": "1.829074",
            "READOUT_NOUL_BIAS": "0",
            "READOUT_INSTR_STYLE": "pyrepr",
            "SHIM_STAGGER": "1",
        }
    )
    command = [
        sys.executable,
        str(shim),
        "--helper",
        str(helper),
        "--model",
        model_dir,
        "--host",
        os.getenv("OPENJEV_HOST", "127.0.0.1"),
        "--port",
        os.getenv("OPENJEV_PORT", "3000"),
    ]
    print("Starting OpenJev at http://127.0.0.1:3000/v1/systemone", flush=True)
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()

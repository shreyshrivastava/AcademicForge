import os
import platform
import importlib.util
from functools import lru_cache


RUNTIME_PROFILE = "MLX+AMD"
APP_VERSION = "academicforge-local-mlx-amd"


@lru_cache(maxsize=1)
def detect_runtime() -> dict:
    system = platform.system()
    machine = platform.machine()
    report = {
        "profile": RUNTIME_PROFILE,
        "app_version": APP_VERSION,
        "system": system,
        "machine": machine,
        "provider": "transformers",
        "accelerator": "cpu",
        "device": "cpu",
        "torch_available": False,
        "rocm_available": False,
        "mlx_available": False,
    }

    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        report["provider"] = "mlx"
        report["accelerator"] = "mlx"
        report["device"] = "Apple Silicon"
        report["mlx_available"] = importlib.util.find_spec("mlx_lm") is not None
        return report

    try:
        import torch

        report["torch_available"] = True
        hip_version = getattr(torch.version, "hip", None)
        cuda_available = bool(torch.cuda.is_available())
        report["rocm_available"] = bool(hip_version and cuda_available)
        report["torch_hip_version"] = hip_version
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            report["device"] = device_name
            lower_name = device_name.lower()
            if hip_version or any(marker in lower_name for marker in ("amd", "radeon", "instinct", "mi2", "mi3")):
                report["accelerator"] = "rocm"
            else:
                report["accelerator"] = "cuda"
        else:
            report["device"] = "cpu"
    except Exception as exc:
        report["torch_error"] = str(exc)

    return report


def auto_provider() -> str:
    provider = os.getenv("LOCAL_LLM_PROVIDER") or os.getenv("LLM_BACKEND") or "auto"
    provider = provider.strip().lower()
    if provider and provider != "auto":
        return provider
    return detect_runtime()["provider"]


def runtime_version_payload(config=None) -> dict:
    runtime = detect_runtime()
    payload = dict(runtime)
    if config is not None:
        payload.update(
            {
                "llm_provider": config.llm_provider,
                "llm_model": config.llm_model,
                "summary_model": config.llm_summary_model,
                "guidance_model": config.llm_guidance_model,
                "research_plan_model": config.llm_research_plan_model,
            }
        )
    return payload

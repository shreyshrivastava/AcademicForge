import logging
import os

logger = logging.getLogger(__name__)


def select_retrieval_device() -> str:
    requested = os.getenv("ACADEMICFORGE_RETRIEVAL_DEVICE", "auto").strip().lower()
    if requested and requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception as exc:
        logger.debug("Could not inspect torch retrieval devices: %s", exc)

    return "cpu"

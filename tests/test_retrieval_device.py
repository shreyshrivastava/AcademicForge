import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.retrieval.device import select_retrieval_device


def test_retrieval_device_env_override():
    original = os.environ.get("ACADEMICFORGE_RETRIEVAL_DEVICE")
    try:
        os.environ["ACADEMICFORGE_RETRIEVAL_DEVICE"] = "cpu"
        assert select_retrieval_device() == "cpu"
    finally:
        if original is None:
            os.environ.pop("ACADEMICFORGE_RETRIEVAL_DEVICE", None)
        else:
            os.environ["ACADEMICFORGE_RETRIEVAL_DEVICE"] = original


if __name__ == "__main__":
    test_retrieval_device_env_override()
    print("retrieval device tests passed")

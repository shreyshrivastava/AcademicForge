"""Optional explicit weight download helper: python -m scripts.download_weights."""

import argparse
import os

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", default=None)
    args = parser.parse_args()
    load_dotenv('.env')
    from backend.config import get_config
    from huggingface_hub import snapshot_download

    config = get_config()
    models = ["BAAI/bge-small-en-v1.5", "BAAI/bge-reranker-base"]
    if config.llm_provider != "fireworks":
        models.append(args.llm or config.llm_model)
    for model in models:
        print(f"Downloading {model}...", flush=True)
        snapshot_download(model, token=os.getenv("HF_TOKEN"))
    print("Weights cached. The backend loads models during startup.")


if __name__ == "__main__":
    main()

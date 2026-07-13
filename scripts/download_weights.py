import os
import argparse
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from backend.retrieval.device import select_retrieval_device

def download_llm(model_name: str, token: str = None):
    print(f"Downloading LLM model: {model_name}...")
    AutoTokenizer.from_pretrained(model_name, token=token)
    snapshot_download(
        repo_id=model_name,
        token=token,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.safetensors",
            "*.txt",
            "tokenizer*",
            "generation_config.json",
        ],
    )
    print("LLM model downloaded successfully.")

def download_embedding(model_name: str):
    device = select_retrieval_device()
    print(f"Downloading Embedding model: {model_name} on {device}...")
    SentenceTransformer(model_name, device=device)
    print("Embedding model downloaded successfully.")

def download_cross_encoder(model_name: str):
    device = select_retrieval_device()
    print(f"Downloading Cross-Encoder model: {model_name} on {device}...")
    CrossEncoder(model_name, device=device)
    print("Cross-Encoder downloaded successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="google/gemma-2-2b-it", help="LLM model name on HuggingFace")
    parser.add_argument("--embed", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model name on HuggingFace")
    parser.add_argument("--reranker", type=str, default="BAAI/bge-reranker-base", help="Cross-Encoder model name")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN environment variable is not set. Gated models like Gemma will fail to download!")

    download_llm(args.llm, token=hf_token)
    download_embedding(args.embed)
    download_cross_encoder(args.reranker)
    print("All weights downloaded and cached successfully!")

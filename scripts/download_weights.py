import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder

def download_llm(model_name: str, token: str = None):
    print(f"Downloading LLM model: {model_name}...")
    AutoTokenizer.from_pretrained(model_name, token=token)
    AutoModelForCausalLM.from_pretrained(model_name, token=token)
    print("LLM model downloaded successfully.")

def download_embedding(model_name: str):
    print(f"Downloading Embedding model: {model_name}...")
    SentenceTransformer(model_name)
    print("Embedding model downloaded successfully.")

def download_cross_encoder(model_name: str):
    print(f"Downloading Cross-Encoder model: {model_name}...")
    CrossEncoder(model_name)
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

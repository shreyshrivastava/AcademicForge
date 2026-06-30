import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.llm import generate_text, model_name, provider_name
from backend.roadmap_generator import generate_roadmap
from backend.summarizer import summarize_paper


SAMPLE_PAPERS = [
    {
        "paper_id": "rag-detector",
        "title": "Probabilistic distances-based hallucination detection in LLMs with RAG",
        "authors": ["Rodion Oblovatny", "Alexandra Kuleshova"],
        "abstract": (
            "Detecting hallucinations in large language models is critical for safety. "
            "This paper proposes an unsupervised RAG hallucination detector based on "
            "distances between prompt token embedding distributions and response token "
            "embedding distributions."
        ),
        "url": "https://example.com/rag",
        "link": "https://example.com/rag",
        "date": "2025-06-11",
        "bm25_rank": 1,
        "dense_rank": 2,
        "rrf_score": 0.032,
    },
    {
        "paper_id": "benchmark",
        "title": "OpenHalDet: A Unified Benchmark for Hallucination Detection",
        "authors": ["Xinyi Li", "Zhen Fang"],
        "abstract": (
            "OpenHalDet standardizes hallucination detection evaluation across diverse "
            "generation scenarios, detector families, tasks, and metrics."
        ),
        "url": "https://example.com/bench",
        "link": "https://example.com/bench",
        "date": "2026-06-05",
        "bm25_rank": 2,
        "dense_rank": 1,
        "rrf_score": 0.031,
    },
]


def run_step(name, fn):
    started = time.perf_counter()
    output = fn()
    elapsed = time.perf_counter() - started
    print(f"\n## {name}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print(_preview(output))
    return output, elapsed


def _preview(text, limit=1400):
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "\n..."


def benchmark(model=None, skip_summary=False):
    if model:
        os.environ["LOCAL_LLM_MODEL"] = model

    print(f"provider={provider_name()}")
    print(f"default_model={model_name()}")
    print(f"summary_model={model_name('summary')}")
    print(f"roadmap_model={model_name('roadmap')}")

    run_step(
        "tiny_generation",
        lambda: generate_text(
            "You are a concise research assistant.",
            "In one sentence, say what a roadmap model should optimize for.",
            token_budget=60,
            task="roadmap",
        ),
    )

    if skip_summary:
        summaries = [
            "Core idea\nDetect RAG hallucinations with hidden-state distribution distances.\n"
            "Method\nCompare prompt and response token embeddings.\n"
            "Implementation notes\nUse as a post-processing detector.\n"
            "Limitations or unknowns\nNeeds hidden-state access.",
            "Core idea\nStandardize hallucination detector evaluation.\n"
            "Method\nUse shared datasets, generation settings, and metrics.\n"
            "Implementation notes\nUse as a benchmark harness.\n"
            "Limitations or unknowns\nCoverage may miss deployment-specific settings.",
        ]
    else:
        summaries = []
        for index, paper in enumerate(SAMPLE_PAPERS, start=1):
            summary, _ = run_step(
                f"summary_{index}",
                lambda paper=paper: summarize_paper(paper),
            )
            summaries.append(summary)

    run_step(
        "roadmap",
        lambda: generate_roadmap(SAMPLE_PAPERS, summaries),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AcademicForge local LLM tasks.")
    parser.add_argument("--model", help="Override LOCAL_LLM_MODEL for this run.")
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Use fixed summaries and benchmark roadmap generation only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark(model=args.model, skip_summary=args.skip_summary)

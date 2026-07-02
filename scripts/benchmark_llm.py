import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.cache as cache
import backend.roadmap_generator as roadmap_generator
import backend.summarizer as summarizer
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

FIXED_SUMMARIES = [
    "One-sentence takeaway\nDetect RAG hallucinations with hidden-state distribution distances.\n"
    "How it works\nCompare prompt and response token embeddings.\n"
    "Why a builder should care\nUse as a post-processing detector.\n"
    "What to verify in the full paper\nNeeds hidden-state access.",
    "One-sentence takeaway\nStandardize hallucination detector evaluation.\n"
    "How it works\nUse shared datasets, generation settings, and metrics.\n"
    "Why a builder should care\nUse as a benchmark harness.\n"
    "What to verify in the full paper\nCoverage may miss deployment-specific settings.",
]


@dataclass
class StepResult:
    name: str
    elapsed_seconds: float
    output_chars: int
    output_words: int
    flags: dict
    preview: str


@dataclass
class ModelRun:
    model: str
    provider: str
    run_index: int
    total_seconds: float
    steps: list[StepResult]


def run_step(name, fn):
    started = time.perf_counter()
    output = fn()
    elapsed = time.perf_counter() - started
    text = str(output).strip()
    return StepResult(
        name=name,
        elapsed_seconds=round(elapsed, 3),
        output_chars=len(text),
        output_words=len(text.split()),
        flags=quality_flags(text),
        preview=_preview(text),
    )


def quality_flags(text):
    lowered = text.strip().lower()
    return {
        "empty": not bool(text.strip()),
        "chatty_preface": lowered.startswith(("okay", "sure", "plain-english", "detailed roadmap")),
        "contains_end_of_turn": "<end_of_turn>" in lowered,
        "mentions_internal_retrieval": any(term in text for term in ("RRF", "BM25=", "dense=")),
    }


def _preview(text, limit=1400):
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "\n..."


def _parse_models(models):
    parsed = []
    for value in models or []:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def _reset_generation_state():
    summarizer.SUMMARY_CACHE.clear()
    roadmap_generator.ROADMAP_CACHE.clear()
    roadmap_generator.PAPER_ROADMAP_CACHE.clear()


class BenchmarkRunner:
    def __init__(self, models=None, runs=1, skip_summary=False, isolated_cache=True):
        self.models = models or [os.getenv("LOCAL_LLM_MODEL") or model_name()]
        self.runs = max(1, runs)
        self.skip_summary = skip_summary
        self.isolated_cache = isolated_cache

    def run(self):
        results = []
        cache_context = tempfile.TemporaryDirectory() if self.isolated_cache else None
        original_cache_dir = cache.CACHE_DIR
        try:
            if cache_context:
                cache.CACHE_DIR = Path(cache_context.name)
            for model in self.models:
                os.environ["LOCAL_LLM_MODEL"] = model
                os.environ["LOCAL_LLM_SUMMARY_MODEL"] = model
                os.environ["LOCAL_LLM_ROADMAP_MODEL"] = model
                for run_index in range(1, self.runs + 1):
                    _reset_generation_state()
                    results.append(self._run_once(model, run_index))
        finally:
            cache.CACHE_DIR = original_cache_dir
            if cache_context:
                cache_context.cleanup()
        return results

    def _run_once(self, model, run_index):
        started = time.perf_counter()
        steps = [
            run_step(
                "tiny_generation",
                lambda: generate_text(
                    "You are a concise research assistant.",
                    "In one sentence, say what a roadmap model should optimize for.",
                    token_budget=60,
                    task="roadmap",
                ),
            )
        ]

        if self.skip_summary:
            summaries = FIXED_SUMMARIES
        else:
            summaries = []
            for index, paper in enumerate(SAMPLE_PAPERS, start=1):
                step = run_step(f"summary_{index}", lambda paper=paper: summarize_paper(paper))
                steps.append(step)
                summaries.append(step.preview)

        steps.append(run_step("roadmap", lambda: generate_roadmap(SAMPLE_PAPERS, summaries)))
        total_seconds = round(time.perf_counter() - started, 3)
        return ModelRun(
            model=model,
            provider=provider_name(),
            run_index=run_index,
            total_seconds=total_seconds,
            steps=steps,
        )


def summarize_results(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result.model, []).append(result)

    summary = []
    for model, runs in grouped.items():
        totals = [run.total_seconds for run in runs]
        step_names = sorted({step.name for run in runs for step in run.steps})
        step_stats = {}
        for step_name in step_names:
            values = [
                step.elapsed_seconds
                for run in runs
                for step in run.steps
                if step.name == step_name
            ]
            step_stats[step_name] = {
                "mean_seconds": round(statistics.mean(values), 3),
                "min_seconds": round(min(values), 3),
                "max_seconds": round(max(values), 3),
            }
        summary.append(
            {
                "model": model,
                "provider": runs[0].provider,
                "runs": len(runs),
                "mean_total_seconds": round(statistics.mean(totals), 3),
                "min_total_seconds": round(min(totals), 3),
                "max_total_seconds": round(max(totals), 3),
                "steps": step_stats,
            }
        )
    return summary


def print_text_report(results):
    for result in results:
        print(f"\n# {result.model} | run {result.run_index} | provider={result.provider}")
        print(f"total_seconds={result.total_seconds:.3f}")
        for step in result.steps:
            flags = ", ".join(name for name, active in step.flags.items() if active) or "none"
            print(f"\n## {step.name}")
            print(
                f"elapsed_seconds={step.elapsed_seconds:.3f} "
                f"chars={step.output_chars} words={step.output_words} flags={flags}"
            )
            print(step.preview)

    print("\n# Summary")
    for item in summarize_results(results):
        print(
            f"{item['model']} | runs={item['runs']} | "
            f"mean_total_seconds={item['mean_total_seconds']:.3f}"
        )


def write_json(path, results):
    payload = {
        "results": [
            {
                **asdict(result),
                "steps": [asdict(step) for step in result.steps],
            }
            for result in results
        ],
        "summary": summarize_results(results),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path, results):
    lines = [
        "# AcademicForge LLM Benchmark",
        "",
        "| Model | Provider | Runs | Mean total seconds | Min | Max |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summarize_results(results):
        lines.append(
            f"| `{item['model']}` | `{item['provider']}` | {item['runs']} | "
            f"{item['mean_total_seconds']:.3f} | {item['min_total_seconds']:.3f} | "
            f"{item['max_total_seconds']:.3f} |"
        )
    lines.append("")
    for result in results:
        lines.append(f"## {result.model} - run {result.run_index}")
        lines.append("")
        for step in result.steps:
            flags = ", ".join(name for name, active in step.flags.items() if active) or "none"
            lines.append(
                f"- `{step.name}`: {step.elapsed_seconds:.3f}s, "
                f"{step.output_words} words, flags: {flags}"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def benchmark(model=None, skip_summary=False):
    models = [model] if model else None
    results = BenchmarkRunner(models=models, skip_summary=skip_summary).run()
    print_text_report(results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AcademicForge local LLM tasks.")
    parser.add_argument("--model", action="append", help="Model to benchmark. Repeat or comma-separate.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per model.")
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Use fixed summaries and benchmark roadmap generation only.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use the normal AcademicForge cache. Default isolates benchmark cache.",
    )
    parser.add_argument("--json-out", help="Write machine-readable benchmark results.")
    parser.add_argument("--markdown-out", help="Write a compact Markdown report.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = BenchmarkRunner(
        models=_parse_models(args.model),
        runs=args.runs,
        skip_summary=args.skip_summary,
        isolated_cache=not args.use_cache,
    )
    benchmark_results = runner.run()
    print_text_report(benchmark_results)
    if args.json_out:
        write_json(args.json_out, benchmark_results)
    if args.markdown_out:
        write_markdown(args.markdown_out, benchmark_results)

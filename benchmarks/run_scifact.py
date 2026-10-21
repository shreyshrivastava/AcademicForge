"""Run the AcademicForge retrieval ablation on BEIR SciFact and NFCorpus.

The runner uses the production retrieval classes directly: BM25Index,
DenseIndex, reciprocal_rank_fusion, and rerank_results. It measures prepared
index latency only: loading models and encoding the corpus happen before timing.

Example:
    python -m benchmarks.run_scifact --download --device cpu
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import zipfile

from backend.retrieval.bm25 import BM25Index, stem_and_remove_stopwords
from backend.retrieval.dense import DenseIndex, MODEL_NAME
from backend.retrieval.reranker import rerank_results
from backend.retrieval.rrf import reciprocal_rank_fusion
from benchmarks.metrics import aggregate, paired_bootstrap_difference, query_metrics


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "scifact": {
        "label": "BEIR SciFact",
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "sha256": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        "documents": 5183,
        "queries": 300,
    },
    "nfcorpus": {
        "label": "BEIR NFCorpus",
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "sha256": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        "documents": 3633,
        "queries": 323,
    },
}
SYSTEMS = (
    "bm25",
    "bm25_stem_stop",
    "bge_dense",
    "rrf",
    "rrf_cross_encoder",
)
METRICS = ("ndcg@10", "recall@10", "mrr@10")


def download(archive: Path, dataset: dict) -> None:
    import requests

    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".part")
    try:
        with requests.get(dataset["url"], timeout=(10, 60), stream=True) as response:
            response.raise_for_status()
            with partial.open("wb") as file:
                for chunk in response.iter_content(65536):
                    file.write(chunk)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != dataset["sha256"]:
            raise ValueError(f"{dataset['label']} archive checksum does not match the recorded version")
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)


def load_dataset(archive: Path, dataset_id: str):
    dataset = DATASETS[dataset_id]
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != dataset["sha256"]:
        raise ValueError(f"{dataset['label']} archive checksum does not match the recorded version")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive_file:
        prefix = f"{dataset_id}/"
        corpus = [json.loads(line) for line in archive_file.read(f"{prefix}corpus.jsonl").splitlines()]
        queries = {
            str(query["_id"]): query["text"]
            for query in (json.loads(line) for line in archive_file.read(f"{prefix}queries.jsonl").splitlines())
        }
        qrels = {}
        rows = csv.DictReader(
            io.StringIO(archive_file.read(f"{prefix}qrels/test.tsv").decode()), delimiter="\t"
        )
        for row in rows:
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])

    if len(corpus) != dataset["documents"] or len(qrels) != dataset["queries"]:
        raise ValueError(
            f"Expected {dataset['documents']} documents and {dataset['queries']} test queries for {dataset['label']}"
        )
    ids = {str(document["_id"]) for document in corpus}
    if len(ids) != len(corpus) or not set(qrels).issubset(queries):
        raise ValueError("Duplicate corpus IDs or missing queries")
    if not all(relevant.issubset(ids) for relevant in qrels.values()):
        raise ValueError("Judgments reference documents outside the corpus")
    papers = [
        {
            "paper_id": str(document["_id"]),
            "title": document["title"],
            "abstract": document["text"],
            "source": f"beir-{dataset_id}",
        }
        for document in corpus
    ]
    return papers, queries, qrels


def source_fingerprint():
    paths = [
        "backend/retrieval/bm25.py",
        "backend/retrieval/dense.py",
        "backend/retrieval/rrf.py",
        "backend/retrieval/reranker.py",
        "benchmarks/run_scifact.py",
        "benchmarks/metrics.py",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def percentile_95(values: list[float]) -> float:
    values = sorted(values)
    return values[max(0, int(len(values) * 0.95) - 1)]


def append_measurement(results, rows, latencies, name, query_id, ranked, relevant, elapsed, k):
    values = query_metrics([result.paper_id for result in ranked[:k]], relevant, k)
    rows[name].append(values)
    latencies[name].append(elapsed * 1000)
    results[name].append(
        {
            "query_id": query_id,
            "ranked_document_ids": [result.paper_id for result in ranked[:k]],
            "metrics": values,
            "latency_ms": elapsed * 1000,
        }
    )


def run_dataset(dataset_id: str, archive: Path, k: int):
    dataset = DATASETS[dataset_id]
    papers, queries, qrels = load_dataset(archive, dataset_id)

    prepared = {}
    started = time.perf_counter()
    prepared["bm25"] = BM25Index(papers)
    bm25_prepare = time.perf_counter() - started

    started = time.perf_counter()
    prepared["bm25_stem_stop"] = BM25Index(papers, analyzer=stem_and_remove_stopwords)
    stem_prepare = time.perf_counter() - started

    started = time.perf_counter()
    prepared["bge_dense"] = DenseIndex(papers)
    dense_prepare = time.perf_counter() - started
    if prepared["bge_dense"].backend != MODEL_NAME:
        raise RuntimeError(
            "BGE dense retrieval is unavailable. Install model dependencies and weights, or choose a usable --device; "
            "no ablation metrics were written."
        )

    per_query, metric_rows, latencies = (
        {name: [] for name in SYSTEMS},
        {name: [] for name in SYSTEMS},
        {name: [] for name in SYSTEMS},
    )
    reranker_backend = "unavailable"
    for query_id in sorted(qrels, key=lambda value: int(value) if value.isdigit() else value):
        query = queries[query_id]
        relevant = qrels[query_id]

        started = time.perf_counter()
        bm25_results = prepared["bm25"].search(query, top_k=50)
        bm25_elapsed = time.perf_counter() - started
        append_measurement(per_query, metric_rows, latencies, "bm25", query_id, bm25_results, relevant, bm25_elapsed, k)

        started = time.perf_counter()
        stem_results = prepared["bm25_stem_stop"].search(query, top_k=50)
        stem_elapsed = time.perf_counter() - started
        append_measurement(
            per_query, metric_rows, latencies, "bm25_stem_stop", query_id, stem_results, relevant, stem_elapsed, k
        )

        started = time.perf_counter()
        dense_results = prepared["bge_dense"].search(query, top_k=50)
        dense_elapsed = time.perf_counter() - started
        append_measurement(
            per_query, metric_rows, latencies, "bge_dense", query_id, dense_results, relevant, dense_elapsed, k
        )

        started = time.perf_counter()
        fused_results = reciprocal_rank_fusion([bm25_results, dense_results], top_k=50)
        fusion_elapsed = time.perf_counter() - started
        rrf_elapsed = bm25_elapsed + dense_elapsed + fusion_elapsed
        append_measurement(per_query, metric_rows, latencies, "rrf", query_id, fused_results, relevant, rrf_elapsed, k)

        started = time.perf_counter()
        reranked_results = rerank_results(query, fused_results, top_k=k)
        rerank_elapsed = time.perf_counter() - started
        reranker_backend = next(
            (result.metadata.get("reranker_backend") for result in reranked_results if result.metadata.get("reranker_backend")),
            reranker_backend,
        )
        append_measurement(
            per_query,
            metric_rows,
            latencies,
            "rrf_cross_encoder",
            query_id,
            reranked_results,
            relevant,
            rrf_elapsed + rerank_elapsed,
            k,
        )

    if reranker_backend != "BAAI/bge-reranker-base":
        raise RuntimeError(
            "BGE cross-encoder reranking is unavailable. Install model dependencies and weights, or choose a usable "
            "--device; no ablation metrics were written."
        )

    results = {}
    for name in SYSTEMS:
        results[name] = {
            "metrics": aggregate(metric_rows[name]),
            "latency_ms": {"p50": statistics.median(latencies[name]), "p95": percentile_95(latencies[name])},
            "per_query": per_query[name],
        }
    for name in SYSTEMS:
        if name == "bm25":
            continue
        results[name]["paired_bootstrap_vs_bm25"] = {
            metric: paired_bootstrap_difference(metric_rows[name], metric_rows["bm25"], metric)
            for metric in METRICS
        }

    return {
        "dataset": dataset["label"],
        "dataset_id": dataset_id,
        "split": "test",
        "dataset_url": dataset["url"],
        "dataset_sha256": dataset["sha256"],
        "documents": len(papers),
        "queries": len(qrels),
        "positive_judgments": sum(map(len, qrels.values())),
        "cutoff": k,
        "prepared_index_seconds": {
            "bm25": bm25_prepare,
            "bm25_stem_stop": stem_prepare,
            "bge_dense": dense_prepare,
            "rrf": max(bm25_prepare, dense_prepare),
            "rrf_cross_encoder": max(bm25_prepare, dense_prepare),
        },
        "systems": results,
        "configuration": {
            "bm25": {"k1": 1.5, "b": 0.75, "text": "title + abstract"},
            "bm25_stem_stop": {
                "analyzer": "simple English suffix stemming + fixed compact stopword list",
                "application_default": False,
            },
            "dense_model": MODEL_NAME,
            "reranker_model": "BAAI/bge-reranker-base",
            "rrf_k": 60,
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 42,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="scifact,nfcorpus", help="Comma-separated: scifact,nfcorpus")
    parser.add_argument("--download", action="store_true", help="Download missing official BEIR archives")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmarks/retrieval-ablation.json")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    os.environ["ACADEMICFORGE_RETRIEVAL_DEVICE"] = args.device

    selected = [dataset.strip() for dataset in args.datasets.split(",") if dataset.strip()]
    if not selected or any(dataset not in DATASETS for dataset in selected):
        parser.error("--datasets must contain one or both of: scifact,nfcorpus")
    reports = []
    for dataset_id in selected:
        archive = ROOT / ".academicforge_cache/benchmarks" / f"{dataset_id}.zip"
        if not archive.exists():
            if not args.download:
                parser.error(f"{DATASETS[dataset_id]['label']} is missing. Use --download.")
            download(archive, DATASETS[dataset_id])
        reports.append(run_dataset(dataset_id, archive, args.k))

    version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {
        "benchmark": "AcademicForge retrieval ablation",
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": version,
        "source_sha256": source_fingerprint(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "retrieval_device_requested": args.device,
        },
        "scope": "Prepared-index retrieval only; excludes model load, corpus encoding, network, generation, and UI.",
        "datasets": reports,
        "limitations": [
            "Metrics use official positive qrels; unjudged documents are treated as nonrelevant.",
            "Intervals and significance tests resample benchmark queries, not domains or users.",
            "The stemmed BM25 row is an ablation and does not change application-default retrieval.",
            "Generation quality, live-source coverage, and factuality are not measured here.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "datasets": reports}, indent=2))


if __name__ == "__main__":
    main()

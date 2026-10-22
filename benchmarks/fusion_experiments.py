"""Cache-first SciFact fusion evaluation.

The official SciFact archive has train and test judgments, but no dev split.
This program selects retrieval settings on train, then evaluates the frozen
selection on test.  Expensive embeddings and lexical score matrices are cached
under the ignored .academicforge_cache directory; all sweeps are NumPy-only.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import zipfile

import numpy as np

from backend.retrieval.bm25 import BM25Index, paper_text, stem_and_remove_stopwords
from backend.retrieval.dense import MODEL_NAME
from backend.retrieval.device import select_retrieval_device
from backend.retrieval.reranker import get_reranker_model, score_pairs_length_sorted
from benchmarks.metrics import aggregate, paired_bootstrap_difference, query_metrics
from benchmarks.run_scifact import DATASETS, ROOT, download, source_fingerprint

CACHE = ROOT / ".academicforge_cache" / "fusion"
OUT = ROOT / "docs" / "benchmarks"
K = 10


def _fingerprint():
    """Include this cache-format and evaluation implementation in provenance."""
    values = source_fingerprint()
    values["benchmarks/fusion_experiments.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return values


def _load(split, dataset_id="scifact"):
    """Read a checksum-verified official BEIR split without retaining abstracts."""
    dataset = DATASETS[dataset_id]
    archive = ROOT / ".academicforge_cache" / "benchmarks" / f"{dataset_id}.zip"
    if not archive.exists():
        download(archive, dataset)
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != dataset["sha256"]:
        raise ValueError(f"{dataset['label']} archive checksum does not match the recorded version")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        corpus = [json.loads(line) for line in z.read(f"{dataset_id}/corpus.jsonl").splitlines()]
        all_queries = {str(row["_id"]): row["text"] for row in
                       (json.loads(line) for line in z.read(f"{dataset_id}/queries.jsonl").splitlines())}
        rows = csv.DictReader(io.StringIO(z.read(f"{dataset_id}/qrels/{split}.tsv").decode()), delimiter="\t")
        qrels = {}
        for row in rows:
            if int(row["score"]) > 0:
                qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    if len(corpus) != dataset["documents"] or not qrels:
        raise ValueError(f"Unexpected {dataset['label']} corpus or empty qrels")
    ids = [str(row["_id"]) for row in corpus]
    if len(ids) != len(set(ids)) or not set(qrels).issubset(all_queries):
        raise ValueError(f"{dataset['label']} identifiers failed validation")
    return corpus, all_queries, qrels


def _ids(qrels, limit=None):
    values = sorted(qrels, key=lambda x: int(x) if x.isdigit() else x)
    if limit is None:
        return values
    # Deterministic prefix is intentional for the smoke test only; it is never a benchmark.
    return values[:limit]


def _bm25_scores(index, query, n):
    tokens = index.analyzer(query)
    if not tokens:
        return np.zeros(n, dtype=np.float32)
    import math
    weights = [(token, math.log(1 + (n - index.doc_freqs[token] + .5) / (index.doc_freqs[token] + .5))) for token in tokens]
    scores = np.zeros(n, dtype=np.float32)
    for pos, (freq, length) in enumerate(zip(index.frequencies, index.lengths)):
        total = 0.0
        for token, idf in weights:
            count = freq[token]
            if count:
                total += idf * (count * 2.5) / (count + 1.5 * (1 - .75 + .75 * length / index.avg_doc_len))
        scores[pos] = total
    return scores


def _cache_split(split, max_queries=None, progress=False, dataset_id="scifact"):
    """Create independently resumable corpus/query/score cache files."""
    CACHE.mkdir(parents=True, exist_ok=True)
    corpus, queries, qrels = _load(split, dataset_id)
    doc_ids = [str(row["_id"]) for row in corpus]
    query_ids = _ids(qrels, max_queries)
    prefix = CACHE / f"{dataset_id}_{split}"
    corpus_path = CACHE / f"{dataset_id}_corpus_bge.npy"
    if not corpus_path.exists():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device=select_retrieval_device())
        np.save(corpus_path, model.encode([f"{p['title']}\n{p['text']}" for p in corpus], normalize_embeddings=True, show_progress_bar=True).astype(np.float32))
    # Use separate names for smoke so its partial arrays never masquerade as full split output.
    suffix = f"_{max_queries}" if max_queries is not None else ""
    query_path = Path(str(prefix) + f"_queries{suffix}.npy")
    dense_path = Path(str(prefix) + f"_dense_scores{suffix}.npy")
    bm25_path = Path(str(prefix) + f"_bm25_scores{suffix}.npy")
    stem_path = Path(str(prefix) + f"_stem_bm25_scores{suffix}.npy")
    ids_path = Path(str(prefix) + f"_query_ids{suffix}.json")
    if not ids_path.exists():
        ids_path.write_text(json.dumps(query_ids))
    elif json.loads(ids_path.read_text()) != query_ids:
        raise ValueError("Cached query identifiers do not match the requested split")
    if not query_path.exists():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device=select_retrieval_device())
        chunks = []
        for start in range(0, len(query_ids), 10):
            chunks.append(model.encode([queries[q] for q in query_ids[start:start + 10]], normalize_embeddings=True).astype(np.float32))
            if progress:
                print(f"[{dataset_id} {split}] encoded {min(start + 10, len(query_ids))}/{len(query_ids)} queries", flush=True)
        np.save(query_path, np.concatenate(chunks))
    corpus_embeddings = np.load(corpus_path, mmap_mode="r")
    query_embeddings = np.load(query_path, mmap_mode="r")
    if not dense_path.exists():
        np.save(dense_path, np.asarray(query_embeddings @ corpus_embeddings.T, dtype=np.float32))
    if not bm25_path.exists() or not stem_path.exists():
        papers = [{"paper_id": str(p["_id"]), "title": p["title"], "abstract": p["text"]} for p in corpus]
        plain, stemmed = BM25Index(papers), BM25Index(papers, analyzer=stem_and_remove_stopwords)
        plain_rows, stem_rows = [], []
        for number, qid in enumerate(query_ids, 1):
            plain_rows.append(_bm25_scores(plain, queries[qid], len(papers)))
            stem_rows.append(_bm25_scores(stemmed, queries[qid], len(papers)))
            if progress and (number % 10 == 0 or number == len(query_ids)):
                print(f"[{dataset_id} {split}] lexical scores {number}/{len(query_ids)}", flush=True)
        np.save(bm25_path, np.asarray(plain_rows, dtype=np.float32))
        np.save(stem_path, np.asarray(stem_rows, dtype=np.float32))
    return {"corpus": corpus, "queries": queries, "qrels": qrels, "doc_ids": doc_ids, "query_ids": query_ids,
            "dense": np.load(dense_path, mmap_mode="r"), "bm25": np.load(bm25_path, mmap_mode="r"),
            "stem": np.load(stem_path, mmap_mode="r"), "cache_files": [str(x) for x in (corpus_path, query_path, dense_path, bm25_path, stem_path, ids_path)]}


def _rank(scores):
    return np.argsort(-scores, axis=1, kind="stable")


def _rrf(left, right, dense_weight=1, k=60):
    n, docs = left.shape
    ranks_left = np.empty((n, docs), dtype=np.int32); ranks_left[np.arange(n)[:, None], _rank(left)] = np.arange(1, docs + 1)
    ranks_right = np.empty((n, docs), dtype=np.int32); ranks_right[np.arange(n)[:, None], _rank(right)] = np.arange(1, docs + 1)
    return dense_weight / (k + ranks_left) + 1.0 / (k + ranks_right)


def _minmax(matrix):
    low, high = matrix.min(axis=1, keepdims=True), matrix.max(axis=1, keepdims=True)
    return (matrix - low) / np.maximum(high - low, 1e-12)


def _rows(scores, data):
    order = _rank(scores)[:, :K]
    return [query_metrics([data["doc_ids"][index] for index in ranked], data["qrels"][qid], K)
            for qid, ranked in zip(data["query_ids"], order)]


def _measure(factory, data):
    values, rows = [], []
    for pos in range(len(data["query_ids"])):
        start = time.perf_counter(); one = factory(pos); values.append((time.perf_counter() - start) * 1000)
        rows.append(query_metrics([data["doc_ids"][x] for x in np.argsort(-one, kind="stable")[:K]], data["qrels"][data["query_ids"][pos]], K))
    return rows, {"p50": statistics.median(values), "p95": sorted(values)[max(0, int(.95 * len(values)) - 1)]}


def _sweep(data):
    results = []
    for weight in (1, 2, 3, 5):
        for k in (10, 30, 60):
            score = _rrf(data["dense"], data["stem"], weight, k)
            results.append({"kind": "weighted_rrf", "dense_weight": weight, "bm25_weight": 1, "rrf_k": k,
                            "ndcg@10": aggregate(_rows(score, data))["ndcg@10"]["mean"]})
    dense, stem = _minmax(data["dense"]), _minmax(data["stem"])
    for alpha in np.arange(0.0, 1.01, .1):
        results.append({"kind": "score_fusion", "dense_alpha": round(float(alpha), 1),
                        "ndcg@10": aggregate(_rows(alpha * dense + (1-alpha) * stem, data))["ndcg@10"]["mean"]})
    # Explicit tie ordering: score fusion is simpler, then lower dense weight / lower RRF k.
    preference = {"score_fusion": 0, "weighted_rrf": 1}
    best = max(results, key=lambda x: (x["ndcg@10"], -preference[x["kind"]], -(x.get("dense_weight", 0)), -(x.get("rrf_k", 0))))
    return results, best


def _rerank(data, base_scores, config, progress=False):
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    path = CACHE / f"scifact_test_rerank_{fingerprint}.npz"
    candidates = _rank(base_scores)[:, :20]
    if path.exists():
        saved = np.load(path)
        if np.array_equal(saved["candidates"], candidates):
            rerank_scores = saved["scores"]
        else:
            raise ValueError("Reranker cache candidates do not match frozen fusion configuration")
    else:
        model = get_reranker_model(); rerank_scores = np.empty(candidates.shape, dtype=np.float32)
        for pos, (qid, indices) in enumerate(zip(data["query_ids"], candidates), 1):
            pairs = [(data["queries"][qid], f"{data['corpus'][index]['title']}\n{data['corpus'][index]['text']}") for index in indices]
            rerank_scores[pos-1] = score_pairs_length_sorted(model, pairs, batch_size=8)
            if progress and (pos % 10 == 0 or pos == len(candidates)):
                print(f"[test rerank] {pos}/{len(candidates)} queries", flush=True)
        np.savez_compressed(path, candidates=candidates, scores=rerank_scores)
    output = np.full(base_scores.shape, -np.inf, dtype=np.float32)
    output[np.arange(len(candidates))[:, None], candidates] = rerank_scores
    return output, str(path)


def _report_systems(test, selected, progress):
    rrf_equal = _rrf(test["dense"], test["stem"], 1, 60)
    rrf_best = _rrf(test["dense"], test["stem"], selected["dense_weight"], selected["rrf_k"])
    score_best = _minmax(test["dense"]) * selected["dense_alpha"] + _minmax(test["stem"]) * (1-selected["dense_alpha"])
    systems = {"dense_only": lambda _: test["dense"][_], "stemmed_bm25": lambda _: test["stem"][_],
               "equal_weight_rrf": lambda _: rrf_equal[_], "best_weighted_rrf": lambda _: rrf_best[_],
               "best_score_fusion": lambda _: score_best[_]}
    rows, payload = {}, {}
    for name, factory in systems.items():
        rows[name], latency = _measure(factory, test)
        payload[name] = {"metrics": aggregate(rows[name]), "latency_ms": latency}
    chosen_name = "best_weighted_rrf" if selected["winner"] == "weighted_rrf" else "best_score_fusion"
    chosen_scores = rrf_best if chosen_name == "best_weighted_rrf" else score_best
    reranked, path = _rerank(test, chosen_scores, selected, progress)
    rows["best_fusion_rerank20"], latency = _measure(lambda x: reranked[x], test)
    payload["best_fusion_rerank20"] = {"metrics": aggregate(rows["best_fusion_rerank20"]), "latency_ms": latency,
                                         "reranker_cache": path}
    base = rows["dense_only"]
    for name, values in rows.items():
        if name != "dense_only":
            comparison = paired_bootstrap_difference(values, base, "ndcg@10")
            comparison["significant"] = not (comparison["bootstrap_95_percent_interval"][0] <= 0 <= comparison["bootstrap_95_percent_interval"][1])
            payload[name]["paired_bootstrap_vs_dense_only_ndcg@10"] = comparison
    rerank_base = rows[chosen_name]
    rerank_comparison = paired_bootstrap_difference(rows["best_fusion_rerank20"], rerank_base, "ndcg@10")
    rerank_comparison["significant"] = not (rerank_comparison["bootstrap_95_percent_interval"][0] <= 0 <= rerank_comparison["bootstrap_95_percent_interval"][1])
    payload["best_fusion_rerank20"]["paired_bootstrap_vs_selected_fusion_ndcg@10"] = rerank_comparison
    return payload


def _report_frozen_nfcorpus(test):
    """Evaluate only the configuration frozen on SciFact; no NFCorpus tuning."""
    rrf_equal = _rrf(test["dense"], test["stem"], 1, 60)
    rrf_frozen = _rrf(test["dense"], test["stem"], 2, 10)
    score_frozen = _minmax(test["dense"]) * .9 + _minmax(test["stem"]) * .1
    factories = {
        "dense_only": lambda index: test["dense"][index],
        "stemmed_bm25": lambda index: test["stem"][index],
        "equal_weight_rrf": lambda index: rrf_equal[index],
        "frozen_weighted_rrf": lambda index: rrf_frozen[index],
        "frozen_score_fusion": lambda index: score_frozen[index],
    }
    rows, systems = {}, {}
    for name, factory in factories.items():
        rows[name], latency = _measure(factory, test)
        systems[name] = {"metrics": aggregate(rows[name]), "latency_ms": latency}
    for name, values in rows.items():
        if name == "dense_only":
            continue
        comparison = paired_bootstrap_difference(values, rows["dense_only"], "ndcg@10")
        comparison["significant"] = not (comparison["bootstrap_95_percent_interval"][0] <= 0 <= comparison["bootstrap_95_percent_interval"][1])
        systems[name]["paired_bootstrap_vs_dense_only_ndcg@10"] = comparison
    return systems


def _run_nfcorpus(phase, max_queries, progress):
    """NFCorpus replication using frozen SciFact settings and binary judged relevance."""
    started = time.perf_counter()
    test = _cache_split("test", max_queries=max_queries, progress=progress, dataset_id="nfcorpus")
    cache_seconds = time.perf_counter() - started
    if phase == "nfcorpus-smoke":
        artifact = {
            "classification": "smoke test, subsample; not a benchmark result",
            "dataset": "BEIR NFCorpus", "split": "official test", "queries": len(test["query_ids"]),
            "dataset_sha256": DATASETS["nfcorpus"]["sha256"], "cache_files": test["cache_files"],
            "stage_seconds": {"cache": cache_seconds, "total": time.perf_counter() - started},
            "source_sha256": _fingerprint(),
        }
        target = OUT / "nfcorpus-frozen-smoke-test.json"
        target.write_text(json.dumps(artifact, indent=2) + "\n")
        print(json.dumps({"output": str(target), "stage_seconds": artifact["stage_seconds"]}))
        return
    if max_queries is not None:
        raise ValueError("A frozen NFCorpus benchmark must use the full test split")
    evaluation_started = time.perf_counter()
    systems = _report_frozen_nfcorpus(test)
    evaluation_seconds = time.perf_counter() - evaluation_started
    report = {
        "benchmark": "AcademicForge NFCorpus frozen SciFact-configuration evaluation",
        "classification": "benchmark result", "dataset": "BEIR NFCorpus", "split": "official test",
        "dataset_sha256": DATASETS["nfcorpus"]["sha256"], "documents": len(test["doc_ids"]),
        "queries": len(test["query_ids"]),
        "frozen_configuration": {"source": "BEIR SciFact train sweep", "weighted_rrf": {"dense_weight": 2, "bm25_weight": 1, "rrf_k": 10}, "score_fusion": {"dense_alpha": .9}},
        "relevance": "NFCorpus graded qrels are converted by the existing metric harness to binary relevance: every score greater than zero is relevant; grade magnitude is not used.",
        "systems": systems,
        "environment": {"platform": platform.platform(), "retrieval_device": select_retrieval_device(), "dense_model": MODEL_NAME, "latency_scope": "cached score-array operations on MPS; excludes model inference and is not comparable to CUDA or ROCm"},
        "bootstrap": {"resamples": 1000, "seed": 42}, "cache_files": test["cache_files"],
        "stage_seconds": {"cache": cache_seconds, "evaluation": evaluation_seconds, "total": time.perf_counter() - started},
        "source_sha256": _fingerprint(),
        "limitations": ["Frozen SciFact configuration; no NFCorpus tuning.", "Abstract retrieval only.", "No cross-encoder reranking.", "No human evaluation."],
    }
    target = OUT / "nfcorpus-frozen-evaluation.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(target), "stage_seconds": report["stage_seconds"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "tune", "evaluate", "all", "nfcorpus-smoke", "nfcorpus-evaluate"), default="all")
    parser.add_argument("--max-queries", type=int, help="Smoke-only deterministic prefix; labels output subsample")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("ACADEMICFORGE_RETRIEVAL_DEVICE", "mps")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase.startswith("nfcorpus-"):
        _run_nfcorpus(args.phase, args.max_queries, args.progress)
        return
    if args.phase == "smoke":
        smoke = _cache_split("train", args.max_queries or 20, args.progress)
        sweep, best = _sweep(smoke)
        artifact = {"classification": "smoke test, subsample; not a benchmark result", "split": "train", "queries": len(smoke["query_ids"]), "best": best, "sweep": sweep, "cache_files": smoke["cache_files"], "source_sha256": _fingerprint()}
        (OUT / "fusion-smoke-test.json").write_text(json.dumps(artifact, indent=2) + "\n")
        print(json.dumps({"output": str(OUT / "fusion-smoke-test.json"), "best": best}))
        return
    train = _cache_split("train", progress=args.progress)
    sweep, best = _sweep(train)
    weighted = max((row for row in sweep if row["kind"] == "weighted_rrf"), key=lambda x: x["ndcg@10"])
    score = max((row for row in sweep if row["kind"] == "score_fusion"), key=lambda x: x["ndcg@10"])
    selected = {"winner": best["kind"], "dense_weight": weighted["dense_weight"], "rrf_k": weighted["rrf_k"], "dense_alpha": score["dense_alpha"], "selection_metric": "nDCG@10", "selection_split": "official SciFact train qrels (SciFact supplies no dev split)"}
    sweep_artifact = {"benchmark": "AcademicForge cache-first fusion sweep", "dataset": "BEIR SciFact", "dataset_sha256": DATASETS["scifact"]["sha256"], "tuning_split": selected["selection_split"], "queries": len(train["query_ids"]), "sweep": sweep, "best_overall": best, "best_weighted_rrf": weighted, "best_score_fusion": score, "frozen_test_configuration": selected, "cache_files": train["cache_files"], "source_sha256": _fingerprint()}
    (OUT / "fusion-sweep.json").write_text(json.dumps(sweep_artifact, indent=2) + "\n")
    if args.phase == "tune":
        print(json.dumps({"output": str(OUT / "fusion-sweep.json"), "selected": selected})); return
    test = _cache_split("test", progress=args.progress)
    systems = _report_systems(test, selected, args.progress)
    report = {"benchmark": "AcademicForge SciFact frozen fusion evaluation", "classification": "benchmark result", "dataset": "BEIR SciFact", "split": "official test", "dataset_sha256": DATASETS["scifact"]["sha256"], "documents": len(test["doc_ids"]), "queries": len(test["query_ids"]), "selection": selected, "systems": systems, "environment": {"platform": platform.platform(), "retrieval_device": select_retrieval_device(), "dense_model": MODEL_NAME, "reranker_model": "BAAI/bge-reranker-base", "latency_scope": "prepared-index per-query retrieval on MPS; not comparable to CUDA or ROCm"}, "bootstrap": {"resamples": 1000, "seed": 42}, "cache_files": test["cache_files"], "source_sha256": _fingerprint(), "limitations": ["SciFact abstracts only.", "No human evaluation.", "SciFact has no dev split; train qrels selected the configuration."]}
    (OUT / "retrieval-fusion-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "retrieval-fusion-evaluation.json"), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate AcademicForge's lexical component on the complete BEIR SciFact test split.

python -m benchmarks.run_scifact --download

No external model, fallback, live source, generated label or paid API is used.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import zipfile

from backend.retrieval.bm25 import BM25Index
from benchmarks.metrics import aggregate, query_metrics

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
SHA256 = '536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165'


def load_dataset(archive):
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('SciFact archive checksum does not match the recorded benchmark version')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        corpus = [json.loads(line) for line in z.read('scifact/corpus.jsonl').splitlines()]
        queries = {str(q['_id']): q['text'] for q in
                   (json.loads(line) for line in z.read('scifact/queries.jsonl').splitlines())}
        qrels = {}
        for row in csv.DictReader(io.StringIO(z.read('scifact/qrels/test.tsv').decode()), delimiter='\t'):
            if int(row['score']) > 0:
                qrels.setdefault(row['query-id'], set()).add(row['corpus-id'])
    if len(corpus) != 5183 or len(qrels) != 300:
        raise ValueError('Expected the full 5,183-document / 300-test-query SciFact benchmark')
    ids = {str(doc['_id']) for doc in corpus}
    if len(ids) != len(corpus) or not set(qrels).issubset(queries):
        raise ValueError('Duplicate corpus IDs or missing queries')
    if not all(relevant.issubset(ids) for relevant in qrels.values()):
        raise ValueError('Judgments reference documents outside the corpus')
    papers = [{'paper_id': str(doc['_id']), 'title': doc['title'], 'abstract': doc['text'],
               'source': 'beir-scifact'} for doc in corpus]
    return papers, queries, qrels


def download(archive):
    import requests
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix('.part')
    try:
        with requests.get(URL, timeout=(10, 30), stream=True) as response:
            response.raise_for_status()
            with partial.open('wb') as file:
                for chunk in response.iter_content(65536):
                    file.write(chunk)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Downloaded archive failed checksum verification')
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)


def source_fingerprint():
    paths = ['backend/retrieval/bm25.py', 'backend/retrieval/models.py',
             'benchmarks/run_scifact.py', 'benchmarks/metrics.py']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def run(archive, output, k=10):
    run_started = datetime.now(timezone.utc).isoformat()
    papers, queries, qrels = load_dataset(archive)
    started = time.perf_counter()
    index = BM25Index(papers)
    index_seconds = time.perf_counter() - started
    rows, per_query, latencies = [], [], []
    for query_id in sorted(qrels, key=int):
        started = time.perf_counter()
        results = index.search(queries[query_id], top_k=k)
        elapsed = (time.perf_counter() - started) * 1000
        ranked_ids = [result.paper_id for result in results]
        values = query_metrics(ranked_ids, qrels[query_id], k)
        rows.append(values)
        latencies.append(elapsed)
        per_query.append({'query_id': query_id, 'query': queries[query_id], 'ranked_document_ids': ranked_ids,
                          'judged_relevant_ids': sorted(qrels[query_id]), 'metrics': values, 'latency_ms': elapsed})
    version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    latency_sorted = sorted(latencies)
    report = {
        'benchmark': 'BEIR SciFact', 'split': 'test', 'scope': 'BM25 component only; full corpus',
        'dataset_url': URL, 'dataset_sha256': SHA256, 'documents': len(papers), 'queries': len(qrels),
        'positive_judgments': sum(map(len, qrels.values())), 'cutoff': k,
        'run_started_utc': run_started,
        'git_head': version, 'source_sha256': source_fingerprint(),
        'environment': {'python': sys.version.split()[0], 'platform': platform.platform()},
        'configuration': {'k1': 1.5, 'b': 0.75, 'text': 'title + abstract', 'stemming': False,
                          'stopword_removal': False, 'learned_models': False, 'fallbacks': False,
                          'test_set_tuning': False, 'seed': 42, 'bootstrap_resamples': 1000},
        'bm25': aggregate(rows),
        'timing': {'index_seconds': index_seconds, 'query_p50_ms': statistics.median(latencies),
                   'query_p95_ms': latency_sorted[max(0, int(len(latencies) * 0.95) - 1)]},
        'limitations': ['Measures the lexical component, not the full hybrid or OpenJev pipeline.',
                        'Does not measure summaries, research plans, factuality, or live-source coverage.',
                        'Scientific claims are not representative of every user research question.',
                        'Unjudged documents count as nonrelevant under this benchmark protocol.',
                        'Timing is local search only, excluding network, model loading and generation.',
                        'Bootstrap intervals describe query resampling, not cross-domain performance.'],
        'per_query': per_query,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('benchmark', 'scope', 'documents', 'queries', 'bm25', 'timing')}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / '.academicforge_cache/benchmarks/scifact.zip')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/benchmarks/scifact-bm25.json')
    args = parser.parse_args()
    if not args.archive.exists():
        if not args.download:
            parser.error('Dataset missing. Use --download or --archive with the official SciFact ZIP.')
        download(args.archive)
    run(args.archive, args.output)


if __name__ == '__main__':
    main()

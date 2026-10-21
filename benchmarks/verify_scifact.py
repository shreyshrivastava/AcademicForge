"""Independently cross-check a SciFact run with the TREC reference evaluator.

uv run --no-project --python 3.14 --with pytrec-eval-terrier==0.5.11 python -m benchmarks.verify_scifact
"""

import csv
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import zipfile

import pytrec_eval

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT / '.academicforge_cache/benchmarks/scifact.zip'
    report_path = ROOT / 'docs/benchmarks/scifact-bm25.json'
    report = json.loads(report_path.read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != report['dataset_sha256']:
        raise ValueError('Dataset checksum differs from the recorded run')
    with zipfile.ZipFile(archive) as z:
        qrels = {}
        for row in csv.DictReader(io.StringIO(z.read('scifact/qrels/test.tsv').decode()), delimiter='\t'):
            if int(row['score']) > 0:
                qrels.setdefault(row['query-id'], {})[row['corpus-id']] = 1
    if {row['query_id'] for row in report['per_query']} != set(qrels):
        raise ValueError('The result must contain every test query')
    cutoff = report['cutoff']
    run = {row['query_id']: {doc: float(cutoff - rank) for rank, doc in enumerate(row['ranked_document_ids'])}
           for row in report['per_query']}
    reference = pytrec_eval.RelevanceEvaluator(qrels, {
        f'ndcg_cut.{cutoff}', f'recall.{cutoff}', f'P.{cutoff}', 'recip_rank',
    }).evaluate(run)
    errors = {}
    for ours, theirs in [('ndcg', 'ndcg_cut'), ('recall', 'recall'), ('precision', 'P'), ('mrr', 'recip_rank')]:
        key = f'{ours}@{cutoff}'
        reference_key = theirs if theirs == 'recip_rank' else f'{theirs}_{cutoff}'
        error = max(abs(row['metrics'][key] - reference[row['query_id']][reference_key]) for row in report['per_query'])
        if error >= 1e-12:
            raise ValueError(f'Metric mismatch: {key}, max absolute error {error}')
        errors[key] = error
    output = {
        'reference_package': 'pytrec-eval-terrier', 'version': version('pytrec-eval-terrier'),
        'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'dataset_sha256': report['dataset_sha256'], 'queries_checked': len(qrels),
        'maximum_per_query_absolute_error': errors,
        'note': 'recip_rank is evaluated on the top-10 run, matching MRR@10. Zero error means matching arithmetic, not perfect retrieval.',
    }
    (ROOT / 'docs/benchmarks/scifact-reference-check.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

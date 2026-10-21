"""Macro-averaged binary relevance metrics with fixed cutoff denominators."""

import math
import random
import statistics


def query_metrics(ranked_ids, relevant_ids, k=10):
    if k < 1 or not relevant_ids:
        raise ValueError('A positive cutoff and at least one judged relevant document are required')
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError('Duplicate document IDs would inflate retrieval metrics')
    retrieved = ranked_ids[:k]
    hits = [int(doc_id in relevant_ids) for doc_id in retrieved]
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1))
    first = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
    return {
        f'ndcg@{k}': dcg / ideal,
        f'recall@{k}': sum(hits) / len(relevant_ids),
        f'mrr@{k}': 1 / first if first else 0.0,
        f'precision@{k}': sum(hits) / k,
        f'hit_rate@{k}': float(bool(sum(hits))),
    }


def aggregate(rows, seed=42, resamples=1000):
    if not rows or resamples < 1:
        raise ValueError('Need query measurements and a positive bootstrap count')
    output = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        rng = random.Random(seed)
        means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples))
        output[key] = {
            'mean': statistics.fmean(values),
            'bootstrap_95_percent_interval': [means[int(resamples * 0.025)], means[min(resamples - 1, int(resamples * 0.975))]],
        }
    return output

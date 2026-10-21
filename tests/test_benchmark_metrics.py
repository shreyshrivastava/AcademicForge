"""Check evaluation arithmetic; these examples are not published benchmark results."""

import math

import pytest

from backend.retrieval.bm25 import BM25Index, bm25_search
from benchmarks.metrics import aggregate, query_metrics


def test_rank_metrics_against_hand_calculated_example():
    result = query_metrics(['wrong', 'relevant-a', 'relevant-b'], {'relevant-a', 'relevant-b', 'missing'}, k=3)
    assert result['recall@3'] == pytest.approx(2 / 3)
    assert result['precision@3'] == pytest.approx(2 / 3)
    assert result['mrr@3'] == pytest.approx(1 / 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert result['ndcg@3'] == pytest.approx(dcg / ideal)


def test_short_results_do_not_inflate_precision():
    result = query_metrics(['relevant'], {'relevant', 'missing'}, k=10)
    assert result['precision@10'] == pytest.approx(0.1)
    assert result['recall@10'] == pytest.approx(0.5)


def test_no_hits_are_scored_as_zero():
    result = query_metrics(['wrong'], {'relevant'}, k=10)
    assert all(value == 0 for value in result.values())


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        query_metrics(['relevant', 'relevant'], {'relevant'}, k=10)


def test_macro_average_counts_failed_queries():
    first = query_metrics(['irrelevant', 'a'], {'a', 'b'}, k=2)
    second = query_metrics([], {'c'}, k=2)
    result = aggregate([first, second], seed=42, resamples=100)
    assert result['recall@2']['mean'] == pytest.approx(0.25)
    assert result['mrr@2']['mean'] == pytest.approx(0.25)
    assert result == aggregate([first, second], seed=42, resamples=100)


def test_prepared_index_retains_keyword_scoring_and_stable_ties():
    papers = [
        {'paper_id': 'wrong', 'title': 'A different subject', 'abstract': 'Unrelated evidence'},
        {'paper_id': 'a', 'title': 'retrieval model', 'abstract': 'research'},
        {'paper_id': 'b', 'title': 'retrieval model', 'abstract': 'research'},
    ]
    prepared = BM25Index(papers).search('retrieval', 10)
    assert [result.paper_id for result in prepared] == ['a', 'b']
    assert [result.metadata['bm25_score'] for result in prepared] == [
        result.metadata['bm25_score'] for result in bm25_search('retrieval', papers, 10)
    ]
    # Independent single-term BM25 calculation; title+abstract length is three.
    expected = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5)) * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 3 / (11 / 3)))
    assert prepared[0].metadata['bm25_score'] == pytest.approx(expected)

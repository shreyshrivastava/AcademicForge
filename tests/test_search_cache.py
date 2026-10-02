import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.data_pipeline as data_pipeline
from backend.search_cache import clear


def test_fast_search_caches_completed_result(monkeypatch):
    clear()
    calls = {"source": 0}
    paper = {
        "paper_id": "cache-paper",
        "title": "Cached retrieval paper",
        "abstract": "A paper about cached retrieval.",
        "authors": [],
        "source": "test",
        "url": "https://example.com/cache-paper",
        "published": "2025-01-01",
    }

    def fake_search(*args, **kwargs):
        calls["source"] += 1
        return [paper], False

    monkeypatch.setattr(data_pipeline, "search_live_candidates", fake_search)
    monkeypatch.setattr(data_pipeline, "filter_relevant_papers", lambda query, papers: papers)
    monkeypatch.setattr(data_pipeline, "rank_papers", lambda query, papers: papers)
    monkeypatch.setattr(data_pipeline, "select_evidence_set", lambda papers, **kwargs: papers)
    monkeypatch.setattr(data_pipeline, "MIN_RELEVANT_PAPERS", 1)
    monkeypatch.setattr(data_pipeline, "FINAL_EVIDENCE_TARGET", 1)

    first = data_pipeline.retrieve_and_rank_papers("cached retrieval")
    second = data_pipeline.retrieve_and_rank_papers("cached retrieval")

    assert calls["source"] == 1
    assert first == second
    assert first is not second

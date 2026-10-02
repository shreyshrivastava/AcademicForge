import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import AppConfig
from backend.jev import enrich_papers_with_jev, rerank_papers_with_jev


def _config(enabled, key="test-key"):
    values = {
        "JEV_ENABLED": "true" if enabled else "false",
        "ACADEMICFORGE_SEARCH_MODE": "quality",
        "TYPESAFE_API_KEY": key,
        "JEV_INCLUDE_DETAILS": "true",
        "JEV_MAX_PAPERS": "3",
    }
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        return AppConfig.from_env()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_jev_disabled_keeps_papers_unchanged():
    papers = [{"paper_id": "p1", "title": "Paper", "abstract": "Text"}]
    assert enrich_papers_with_jev("query", papers, _config(False)) == papers


def test_jev_response_is_normalized(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "answers": {
                    "paper_relevance": {
                        "choice": "paper_0",
                        "confidence": 0.9,
                        "probabilities": {"paper_0": 1.0},
                    },
                    "paper_0_type": {"choice": "method", "confidence": 0.8},
                    "paper_0_difficulty": {"choice": "medium", "confidence": 0.7},
                }
            }

    monkeypatch.setattr("backend.jev.requests.post", lambda *args, **kwargs: Response())
    result = enrich_papers_with_jev(
        "query",
        [{"paper_id": "p1", "title": "Paper", "abstract": "Text"}],
        _config(True),
    )
    assert result[0]["metadata"]["jev"]["relevance_score"] == 1.0
    assert result[0]["metadata"]["jev"]["paper_type"] == "method"


def test_local_openjev_does_not_need_api_key(monkeypatch):
    monkeypatch.setenv("JEV_ENABLED", "true")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("JEV_ENDPOINT", "http://127.0.0.1:3000/v1/systemone")
    config = AppConfig.from_env()
    assert enrich_papers_with_jev("query", [], config) == []


def test_fast_mode_disables_openjev(monkeypatch):
    monkeypatch.setenv("ACADEMICFORGE_SEARCH_MODE", "fast")
    monkeypatch.setenv("JEV_ENABLED", "true")
    monkeypatch.setenv("JEV_ENDPOINT", "http://127.0.0.1:3000/v1/systemone")
    config = AppConfig.from_env()
    assert config.search_mode == "fast"
    assert not config.jev_enabled


def test_jev_relevance_reorders_search_results(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "answers": {
                    "paper_relevance": {
                        "choice": "paper_1",
                        "probabilities": {"paper_0": 0.2, "paper_1": 0.9},
                    }
                }
            }

    monkeypatch.setattr("backend.jev.requests.post", lambda *args, **kwargs: Response())
    papers = [
        {"paper_id": "p1", "title": "Keyword match", "abstract": "Text"},
        {"paper_id": "p2", "title": "Better answer", "abstract": "Text"},
    ]
    result = rerank_papers_with_jev("query", papers, _config(True))
    assert [paper["paper_id"] for paper in result] == ["p2", "p1"]
    assert result[0]["metadata"]["jev"]["relevance_score"] == 0.9


def test_jev_rerank_falls_back_to_existing_order(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.RequestException("offline")

    monkeypatch.setattr("backend.jev.requests.post", fail)
    papers = [
        {"paper_id": "p1", "title": "First", "abstract": "Text"},
        {"paper_id": "p2", "title": "Second", "abstract": "Text"},
    ]
    result = rerank_papers_with_jev("query", papers, _config(True))
    assert [paper["paper_id"] for paper in result] == ["p1", "p2"]


def test_jev_caches_successful_search_ranking(monkeypatch):
    calls = 0

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"answers": {"paper_relevance": {"probabilities": {"paper_0": 0.8}}}}

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr("backend.jev.requests.post", post)
    papers = [{"paper_id": "cache-p1", "title": "Cache test", "abstract": "Text"}]
    config = _config(True)
    first = enrich_papers_with_jev("cache query", papers, config)
    second = enrich_papers_with_jev("cache query", papers, config)
    assert calls == 1
    assert first == second

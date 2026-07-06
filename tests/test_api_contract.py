import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import backend.app as backend_app
from backend.app import app


def test_config_endpoint_returns_model_info():
    client = TestClient(app)
    response = client.get("/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_provider"]
    assert payload["llm_models"]["default"]
    assert payload["llm_models"]["summary"]
    assert payload["llm_models"]["roadmap"]
    assert payload["generation_modes"]["fast"]["model"]
    assert payload["generation_modes"]["deep"]["model"]


def test_search_endpoint_passes_research_focus():
    original_retrieve = backend_app.retrieve_and_rank_papers

    def fake_retrieve(query, categories=None):
        assert query == "rag implementation"
        assert categories == ["Implementation Focused"]
        return [
            {
                "title": "Paper",
                "authors": [],
                "abstract": "Abstract",
                "link": "https://example.com",
                "date": "2026-06-30",
                "metadata": {"academicforge_category": "Implementation Focused"},
            }
        ]

    backend_app.retrieve_and_rank_papers = fake_retrieve
    try:
        client = TestClient(app)
        response = client.post(
            "/search",
            json={"query": "rag implementation", "categories": ["Implementation Focused"]},
        )
        assert response.status_code == 200
        assert response.json()["papers"][0]["metadata"]["academicforge_category"] == "Implementation Focused"
    finally:
        backend_app.retrieve_and_rank_papers = original_retrieve


def test_search_endpoint_treats_balanced_as_exclusive():
    original_retrieve = backend_app.retrieve_and_rank_papers

    def fake_retrieve(query, categories=None):
        assert query == "rag implementation"
        assert categories == []
        return []

    backend_app.retrieve_and_rank_papers = fake_retrieve
    try:
        client = TestClient(app)
        response = client.post(
            "/search",
            json={
                "query": "rag implementation",
                "categories": ["Balanced", "Implementation Focused"],
            },
        )
        assert response.status_code == 200
        assert response.json()["papers"] == []
    finally:
        backend_app.retrieve_and_rank_papers = original_retrieve


def test_roadmap_cache_status_endpoint_returns_status():
    client = TestClient(app)
    response = client.post(
        "/roadmap/cache-status",
        json={
            "papers": [
                {
                    "title": "Paper",
                    "authors": [],
                    "abstract": "Abstract",
                    "link": "https://example.com",
                    "date": "2026-06-30",
                }
            ],
            "summaries": ["Core idea\nA test."],
            "query": "build a prototype",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "cached" in payload
    assert payload["cache"] in {"memory", "disk", "miss"}


def test_paper_roadmap_endpoint_returns_roadmap():
    original_generate = backend_app.generate_ai_paper_roadmap

    def fake_generate(paper, model=None):
        assert model == backend_app.DEEP_MODE_MODEL
        return f"roadmap for {paper['title']}"

    backend_app.generate_ai_paper_roadmap = fake_generate
    try:
        client = TestClient(app)
        response = client.post(
            "/roadmap/paper",
            json={
                "title": "Paper",
                "authors": [],
                "abstract": "Abstract",
                "link": "https://example.com",
                "date": "2026-06-30",
                "generation_mode": "deep",
            },
        )
        assert response.status_code == 200
        assert response.json() == {"roadmap": "roadmap for Paper"}
    finally:
        backend_app.generate_ai_paper_roadmap = original_generate


def test_paper_roadmap_cache_status_endpoint_returns_status():
    client = TestClient(app)
    response = client.post(
        "/roadmap/paper/cache-status",
        json={
            "title": "Paper",
            "authors": [],
            "abstract": "Abstract",
            "link": "https://example.com",
            "date": "2026-06-30",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "cached" in payload
    assert payload["cache"] in {"memory", "disk", "miss"}


def test_roadmap_stream_endpoint_streams_text():
    original_stream = backend_app.stream_ai_roadmap

    def fake_stream(papers, summaries=None, query="", model=None):
        assert query == "build a prototype"
        assert model == backend_app.DEEP_MODE_MODEL
        yield "hello "
        yield "roadmap"

    backend_app.stream_ai_roadmap = fake_stream
    try:
        client = TestClient(app)
        with client.stream(
            "POST",
            "/roadmap/stream",
            json={
                "papers": [
                    {
                        "title": "Paper",
                        "authors": [],
                        "abstract": "Abstract",
                        "link": "https://example.com",
                        "date": "2026-06-30",
                    }
                ],
                "summaries": ["Core idea\nA test."],
                "query": "build a prototype",
                "generation_mode": "deep",
            },
        ) as response:
            assert response.status_code == 200
            assert response.read().decode("utf-8") == "hello roadmap"
    finally:
        backend_app.stream_ai_roadmap = original_stream


if __name__ == "__main__":
    test_config_endpoint_returns_model_info()
    test_search_endpoint_passes_research_focus()
    test_roadmap_cache_status_endpoint_returns_status()
    test_paper_roadmap_endpoint_returns_roadmap()
    test_paper_roadmap_cache_status_endpoint_returns_status()
    test_roadmap_stream_endpoint_streams_text()
    print("api contract tests passed")

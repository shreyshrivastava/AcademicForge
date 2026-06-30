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
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "cached" in payload
    assert payload["cache"] in {"memory", "disk", "miss"}


def test_roadmap_stream_endpoint_streams_text():
    original_stream = backend_app.stream_ai_roadmap

    def fake_stream(papers, summaries=None):
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
            },
        ) as response:
            assert response.status_code == 200
            assert response.read().decode("utf-8") == "hello roadmap"
    finally:
        backend_app.stream_ai_roadmap = original_stream


if __name__ == "__main__":
    test_config_endpoint_returns_model_info()
    test_roadmap_cache_status_endpoint_returns_status()
    test_roadmap_stream_endpoint_streams_text()
    print("api contract tests passed")

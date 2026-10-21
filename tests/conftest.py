"""Unit tests never read developer credentials or load/download model weights."""

import os

os.environ["ACADEMICFORGE_ENV_FILE"] = "/nonexistent/academicforge-test.env"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ACADEMICFORGE_REQUESTS_PER_MINUTE"] = "10000"

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("LOCAL_LLM_", "REMOTE_LLM_", "JEV_", "FIREWORKS_")) or key in {
            "GROQ_API_KEY", "TYPESAFE_API_KEY", "ACADEMICFORGE_API_KEY", "HF_TOKEN",
        }:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LOCAL_LLM_PROVIDER", "transformers")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "test/local-model")
    monkeypatch.setenv("ACADEMICFORGE_SEARCH_MODE", "fast")
    monkeypatch.setenv("ACADEMICFORGE_DEMO_MODEL_ENABLED", "false")
    monkeypatch.setattr("backend.retrieval.dense._load_sentence_transformer", lambda: None)

    def unavailable():
        raise RuntimeError("Model loading disabled in unit tests")

    monkeypatch.setattr("backend.retrieval.reranker.get_reranker_model", unavailable)
    from backend.search_cache import clear
    import backend.jev as jev
    from backend.sources import begin_search

    clear()
    jev._JEV_CACHE.clear()
    begin_search()

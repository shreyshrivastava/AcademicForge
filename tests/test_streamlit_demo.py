"""Standalone contracts run in both the full and model-free environments."""

from pathlib import Path

import pytest
import requests

from backend import data_pipeline as pipeline
from backend.search_cache import clear
from backend.sources import source_status
from frontend import demo_runtime as demo

AI = {"paper_id": "ai", "title": "RAGAS: Retrieval Augmented Generation Evaluation",
      "abstract": "Evaluating retrieval and answer generation in LLM systems.",
      "authors": [], "source": "arxiv", "url": "https://arxiv.org/abs/2309.15217", "metadata": {}}
MUSIC = {**AI, "paper_id": "music", "title": "Ragas in music", "abstract": "Melody in Indian classical music."}


@pytest.fixture(autouse=True)
def reset_budget(monkeypatch):
    demo._search_times.clear()
    # The entry point sets this variable; monkeypatch restores it after each UI test.
    monkeypatch.setenv("ACADEMICFORGE_FRONTEND_MODE", "")
    yield
    demo._search_times.clear()


def test_demo_shared_search_is_field_aware_and_never_loads_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Standalone search must not load models, read generation config, or call paid APIs")
    monkeypatch.setattr(pipeline, "get_config", forbidden)
    monkeypatch.setattr(pipeline, "rank_papers", forbidden)
    monkeypatch.setattr(pipeline, "enrich_papers_with_jev", forbidden)
    monkeypatch.setattr(requests, "post", forbidden)
    queries = []
    def source(query, **kwargs):
        queries.append(query)
        return [AI, MUSIC], False
    monkeypatch.setattr(pipeline, "search_live_candidates", source)
    result = demo.search({"query": "rag as", "domain": "Computer science & AI"})
    assert [p["paper_id"] for p in result["papers"]] == ["ai"]
    assert all("retrieval augmented generation evaluation" in q for q in queries)
    assert result["papers"][0]["metadata"]["dense_backend"] == "disabled"
    assert result["capabilities"]["ranking"] == "bm25"


def test_outage_preserves_one_real_paper_with_warning(monkeypatch):
    def partial(query, **kwargs):
        source_status("semantic_scholar", "unavailable")
        return [AI], False
    monkeypatch.setattr(pipeline, "search_live_candidates", partial)
    result = demo.search({"query": "retrieval augmented generation"})
    assert result["papers"][0]["paper_id"] == "ai"
    assert result["warnings"]


def test_all_sources_unavailable_never_invents_evidence(monkeypatch):
    def unavailable(query, **kwargs):
        source_status("arxiv", "unavailable")
        source_status("semantic_scholar", "unavailable")
        return [], False
    monkeypatch.setattr(pipeline, "search_live_candidates", unavailable)
    result = demo.search({"query": "retrieval"})
    assert result["papers"] == []
    assert "unavailable" in result["message"]
    assert len(result["warnings"]) == 2


def test_direct_missing_id_has_a_message(monkeypatch):
    monkeypatch.setattr(pipeline, "search_live_candidates", lambda *a, **k: ([], True))
    assert demo.search({"query": "9999.99999"})["message"]


def test_demo_cache_separates_fields_and_ranking_profiles(monkeypatch):
    calls = []
    def source(query, **kwargs):
        calls.append(query)
        return [AI, MUSIC], False
    monkeypatch.setattr(pipeline, "search_live_candidates", source)
    first = demo.search({"query": "ragas", "domain": "Computer science & AI"})
    assert first["papers"]
    count = len(calls)
    demo.search({"query": "ragas", "domain": "Computer science & AI"})
    assert len(calls) == count
    music = demo.search({"query": "ragas", "domain": "Music & arts"})
    assert music["papers"][0]["paper_id"] == "music"
    assert len(calls) > count
    clear()


def test_repeated_plans_have_provenance_and_do_not_consume_a_lifetime_quota(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "unused-test-key")
    monkeypatch.setenv("ACADEMICFORGE_CLOUD_ENABLE_LLM", "true")
    client = demo.DemoClient()
    for _ in range(2):
        text = "".join(client.stream_post("/research-plan/stream", {
            "papers": [AI], "query": "RAGAS", "domain": "Computer science & AI",
        }))
        assert "Search field: Computer science & AI" in text
        assert "https://arxiv.org/abs/2309.15217" in text
        assert "No AI model" in text
        assert client.last_generation["preview"]
    assert "no ai model" in client.post_json("/paper-guidance", AI)["guidance"].casefold()
    assert "no AI model" in client.post_json("/summarize", AI)["summary"]


def test_process_search_budget_resets(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(demo.time, "monotonic", lambda: clock[0])
    for _ in range(40):
        demo._admit_search()
    with pytest.raises(requests.HTTPError, match="busy"):
        demo._admit_search()
    clock[0] += 61
    demo._admit_search()


def test_concurrent_search_admission_is_bounded():
    for _ in range(3):
        assert demo._search_capacity.acquire(blocking=False)
    try:
        with pytest.raises(requests.HTTPError, match="busy"):
            demo.search({"query": "retrieval"})
    finally:
        for _ in range(3):
            demo._search_capacity.release()


def test_shared_ui_demo_search_preview_and_export(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(pipeline, "search_live_candidates", lambda *a, **k: ([AI, MUSIC], False))
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend/streamlit_cloud_app.py"))
    page.run(timeout=20)
    assert not page.exception
    assert page.radio[0].options == ["Evidence analysis preview"]
    page.selectbox(key="search_field").select("Computer science & AI")
    page.text_input[0].input("rag as")
    next(b for b in page.button if b.label == "Search").click().run(timeout=20)
    assert not page.exception
    assert len(page.session_state.papers) == 1
    assert page.session_state.last_search_domain == "Computer science & AI"
    next(b for b in page.button if b.label == "Guidance").click().run(timeout=20)
    assert not page.exception
    for _ in range(2):
        next(b for b in page.button if b.label == "Generate Research Plan").click().run(timeout=20)
        assert not page.exception
        assert not page.error
        assert "No AI model" in page.session_state.research_plan
    assert len(page.get("download_button")) == 1


def test_full_profile_also_keeps_small_valid_evidence_sets(monkeypatch):
    monkeypatch.setattr(pipeline, "search_live_candidates", lambda *a, **k: ([AI], False))
    monkeypatch.setattr(pipeline, "rank_papers", lambda query, papers: papers)
    assert pipeline.retrieve_and_rank_papers("retrieval augmented generation")[0]["paper_id"] == "ai"

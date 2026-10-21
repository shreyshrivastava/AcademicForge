"""Failure-injection regression tests for the engineering diagnostic."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import requests
from fastapi.testclient import TestClient

import backend.app as api
from backend.config import get_config
from backend.errors import CapacityError, SourceUnavailable, StreamInterrupted
from backend.execution import Capacity, execute
import backend.health as health
import backend.llm as llm
import backend.jev as jev
import backend.data_pipeline as pipeline
from backend.sources import begin_search, diagnostics, source_status
from frontend.api_client import APIClient, StreamError
from frontend.workflow import paper_output_key

PAPER = {"title": "Evidence", "abstract": "Useful evidence", "authors": [], "link": "https://example.com", "date": "2026"}


def remote(monkeypatch):
    monkeypatch.setenv("REMOTE_LLM_USE_FOR_DEEP", "true")
    monkeypatch.setenv("REMOTE_LLM_MODEL", "test/remote")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def test_cloud_key_does_not_override_opt_out(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setenv("LOCAL_LLM_USE_FIREWORKS", "false")
    assert get_config().llm_research_plan_model == "test/local-model"
    cloud = Mock(side_effect=AssertionError("Unauthorized cloud call"))
    monkeypatch.setattr(llm, "_generate_fireworks", cloud)
    with pytest.raises(llm.LocalLLMError):
        llm.LLMService().generate("s", "u", model="accounts/fireworks/test")
    cloud.assert_not_called()


def test_explicit_cloud_model_rejected_when_disabled(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_RESEARCH_PLAN_MODEL", "accounts/fireworks/test")
    with pytest.raises(ValueError, match="explicit opt-in"):
        get_config()


@pytest.mark.parametrize("setting,value", [("LOCAL_LLM_MAX_TOKENS", "0"), ("LOCAL_LLM_TEMPERATURE", "nan"),
                                           ("JEV_TIMEOUT_SECONDS", "-1")])
def test_invalid_config_rejected(monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError):
        get_config()


def test_configuration_reads_values_after_env_change(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MAX_TOKENS", "123")
    assert get_config().llm_max_tokens == 123


@pytest.mark.parametrize("provider", ["remote", "fireworks"])
def test_cloud_failure_falls_back_before_output(monkeypatch, provider):
    if provider == "remote":
        remote(monkeypatch)
        model = "test/remote"
    else:
        monkeypatch.setenv("LOCAL_LLM_USE_FIREWORKS", "true")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
        model = "accounts/fireworks/test"
    monkeypatch.setattr(llm, f"_generate_{provider}", Mock(side_effect=RuntimeError("outage")))
    monkeypatch.setattr(llm, f"_generate_{provider}_stream", Mock(side_effect=RuntimeError("outage")))
    monkeypatch.setattr(llm, "_generate_transformers", lambda *args: "local answer")
    assert llm.LLMService().generate("s", "u", model=model) == "local answer"
    assert list(llm.LLMService().stream("s", "u", model=model)) == ["local answer"]


def test_partial_stream_never_appends_fallback(monkeypatch):
    remote(monkeypatch)
    def fail(*args):
        yield "partial"
        raise RuntimeError("outage")
    monkeypatch.setattr(llm, "_generate_remote_stream", fail)
    local = Mock(return_value="new full answer")
    monkeypatch.setattr(llm, "_generate_transformers", local)
    stream = llm.LLMService().stream("s", "u", model="test/remote")
    assert next(stream) == "partial"
    with pytest.raises(StreamInterrupted):
        next(stream)
    local.assert_not_called()


def test_stream_endpoint_reports_error_and_releases_capacity(monkeypatch):
    def fail(*args, **kwargs):
        yield "partial"
        raise RuntimeError("secret internal details")
    monkeypatch.setattr(api, "stream_ai_research_plan", fail)
    client = TestClient(api.app)
    for _ in range(2):
        response = client.post("/research-plan/stream", json={"papers": [PAPER]})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines()]
        assert [event["type"] for event in events] == ["start", "delta", "error"]
        assert "secret internal details" not in response.text


@pytest.mark.parametrize("path,payload", [
    ("/search", {"query": "  "}),
    ("/search", {"query": "a" * 2001}),
    ("/search", {"query": "test", "categories": ["unknown"]}),
    ("/research-plan", {"papers": []}),
    ("/research-plan", {"papers": [PAPER], "generation_mode": "unknown"}),
    ("/research-plan", {"papers": [PAPER] * 11}),
    ("/research-plan", {"papers": [PAPER], "summaries": ["x", "y"]}),
])
def test_invalid_input_rejected_before_work(monkeypatch, path, payload):
    search = Mock(side_effect=AssertionError("Search should not run"))
    generate = Mock(side_effect=AssertionError("Generation should not run"))
    monkeypatch.setattr(api, "retrieve_and_rank_papers", search)
    monkeypatch.setattr(api, "generate_ai_research_plan", generate)
    assert TestClient(api.app).post(path, json=payload).status_code == 422
    search.assert_not_called()
    generate.assert_not_called()


def test_health_stays_responsive_and_capacity_is_bounded(monkeypatch):
    entered, release = Event(), Event()
    def blocking(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return "answer"
    monkeypatch.setattr(api, "summarize_paper_with_ai", blocking)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            work = asyncio.create_task(client.post("/summarize", json=PAPER))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                live = await asyncio.wait_for(client.get("/live"), 0.5)
                assert live.status_code == 200
                busy = await client.post("/summarize", json=PAPER)
                assert busy.status_code == 429
            finally:
                release.set()
                assert (await work).status_code == 200
    asyncio.run(scenario())


def test_capacity_recovers_after_exception():
    capacity = Capacity(1)
    def fail():
        raise RuntimeError("failure")
    async def scenario():
        with pytest.raises(RuntimeError):
            await execute(capacity, fail)
        assert await execute(capacity, lambda: "ok") == "ok"
    asyncio.run(scenario())


def test_readiness_reports_degraded_retrieval(monkeypatch):
    monkeypatch.setattr(llm, "_load_transformers_model", lambda *args: True)
    health.warmup()
    state = health.readiness()
    assert state["models_ready"] is True
    assert state["status"] == "degraded"
    assert state["components"]["reranker"] == "unavailable"


def test_readiness_fails_when_generation_unavailable(monkeypatch):
    monkeypatch.setattr(llm, "_load_transformers_model", Mock(side_effect=RuntimeError("no model")))
    health.warmup()
    assert health.readiness()["models_ready"] is False
    assert TestClient(api.app).get("/health").status_code == 503


def test_lifespan_warms_and_cleans_up(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "warmup", lambda: calls.append("warm"))
    with TestClient(api.app) as client:
        assert client.get("/live").status_code == 200
    assert calls == ["warm"]


@pytest.mark.parametrize("payload", [[], None, {"answers": []}, {"answers": {"paper_relevance": None}},
                                      {"answers": {"paper_relevance": {"probabilities": {"paper_0": 5}}}}])
def test_malformed_jev_preserves_ranking(monkeypatch, payload):
    config = replace(get_config(), search_mode="quality", jev_enabled=True,
                     jev_endpoint="http://127.0.0.1:3000/v1/systemone")
    response = Mock()
    response.json.return_value = payload
    monkeypatch.setattr(jev.requests, "post", lambda *args, **kwargs: response)
    papers = [dict(PAPER)]
    assert jev.enrich_papers_with_jev("query", papers, config) == papers
    assert diagnostics()["warnings"]


def test_total_source_outage_is_not_no_results(monkeypatch):
    def arxiv(*args, **kwargs):
        source_status("arxiv", "unavailable")
        return [], False
    def semantic(*args, **kwargs):
        source_status("semantic_scholar", "unavailable")
        return []
    monkeypatch.setattr(pipeline, "search_arxiv_candidates", arxiv)
    monkeypatch.setattr(pipeline, "search_semantic_scholar_candidates", semantic)
    with pytest.raises(SourceUnavailable):
        pipeline.search_live_candidates("test")
    response = TestClient(api.app).post("/search", json={"query": "test"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "sources_unavailable"


def test_partial_source_results_remain_available(monkeypatch):
    def arxiv(*args, **kwargs):
        source_status("arxiv", "unavailable")
        return [], False
    def semantic(*args, **kwargs):
        source_status("semantic_scholar", "ok")
        return [PAPER]
    monkeypatch.setattr(pipeline, "search_arxiv_candidates", arxiv)
    monkeypatch.setattr(pipeline, "search_semantic_scholar_candidates", semantic)
    result, direct = pipeline.search_live_candidates("test")
    assert result == [PAPER]
    assert diagnostics()["sources"] == {"arxiv": "unavailable", "semantic_scholar": "ok"}


def test_errors_do_not_echo_internal_details(monkeypatch):
    monkeypatch.setattr(api, "retrieve_and_rank_papers", Mock(side_effect=RuntimeError("secret-password")))
    response = TestClient(api.app).post("/search", json={"query": "test"})
    assert response.status_code == 500
    assert "secret-password" not in response.text
    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]


def test_api_key_protects_work_but_not_liveness(monkeypatch):
    monkeypatch.setenv("ACADEMICFORGE_API_KEY", "test-secret")
    client = TestClient(api.app)
    assert client.get("/live").status_code == 200
    assert client.get("/config").status_code == 401
    assert client.get("/config", headers={"X-API-Key": "test-secret"}).status_code == 200


def test_request_body_limit():
    response = TestClient(api.app).post("/search", content="x" * 1048577)
    assert response.status_code == 413


def client_with_session(session):
    client = APIClient.__new__(APIClient)
    client.base_url = "http://test"
    client.session = session
    client.timeout_generation = 1
    client.max_retries = 3
    return client


def test_generation_post_is_not_retried():
    session = Mock()
    session.request.side_effect = requests.Timeout("timeout")
    with pytest.raises(requests.Timeout):
        client_with_session(session).post_json("/summarize", PAPER)
    assert session.request.call_count == 1


def test_client_requires_stream_completion_and_does_not_retry():
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_lines.return_value = iter(['{"type":"delta","text":"partial"}'])
    session = Mock()
    session.post.return_value = response
    stream = client_with_session(session).stream_post("/research-plan/stream", {})
    assert next(stream) == "partial"
    with pytest.raises(StreamError):
        next(stream)
    assert session.post.call_count == 1


def test_cache_identity_changes_with_model_or_evidence():
    key = paper_output_key(PAPER, "fast", {"summary": "a"})
    assert key != paper_output_key(PAPER, "fast", {"summary": "b"})
    assert key != paper_output_key({**PAPER, "abstract": "changed"}, "fast", {"summary": "a"})


def test_weight_download_module_help_works():
    result = subprocess.run([sys.executable, "-m", "scripts.download_weights", "--help"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--llm" in result.stdout


def test_client_accepts_only_a_complete_stream():
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_lines.return_value = iter([
        '{"type":"start","request_id":"id"}',
        '{"type":"delta","text":"answer"}',
        '{"type":"done","generation":{"fallback":true}}',
    ])
    session = Mock()
    session.post.return_value = response
    client = client_with_session(session)
    assert list(client.stream_post('/research-plan/stream', {})) == ['answer']
    assert client.last_generation['fallback'] is True


def test_cloud_fallback_reports_actual_provider(monkeypatch):
    remote(monkeypatch)
    monkeypatch.setattr(llm, '_generate_remote', Mock(side_effect=RuntimeError('offline')))
    monkeypatch.setattr(llm, '_generate_transformers', lambda *args: 'local answer')
    result = api.generated('summary', llm.generate_text, 's', 'u', model='test/remote')
    assert result['generation'] == {'provider': 'transformers', 'model': 'test/local-model', 'fallback': True}


def test_cancelled_request_does_not_free_running_inference_slot():
    entered, release = Event(), Event()
    capacity = Capacity(1)
    def block():
        entered.set()
        release.wait(3)
    async def scenario():
        task = asyncio.create_task(execute(capacity, block))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        try:
            with pytest.raises(CapacityError):
                capacity.reserve()
        finally:
            release.set()
        for _ in range(100):
            await asyncio.sleep(0.01)
            try:
                lease = capacity.reserve()
            except CapacityError:
                continue
            lease.release()
            return
        pytest.fail('Capacity did not recover')
    asyncio.run(scenario())


def test_launcher_exits_when_backend_dies(tmp_path):
    # Run the real launcher with a controlled interpreter; no model/server subprocess starts.
    source = Path(__file__).resolve().parents[1] / 'start.sh'
    launcher = tmp_path / 'start.sh'
    launcher.write_text(source.read_text())
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/bin/bash\nif [ "$1" = "-" ]; then\n cat >/dev/null\n echo "127.0.0.1 1"\n exit 0\nfi\nexit 7\n')
    fake.chmod(0o755)
    curl = tmp_path / 'curl'
    curl.write_text('#!/bin/bash\nsleep 0.1\nexit 1\n')
    curl.chmod(0o755)
    env = {**os.environ, 'PYTHON_BIN': str(fake), 'PATH': str(tmp_path) + ':' + os.environ['PATH']}
    result = subprocess.run(['/bin/bash', str(launcher)], env=env, capture_output=True, text=True, timeout=8)
    assert result.returncode == 1
    assert 'Backend exited during startup' in result.stderr


def test_launcher_cleans_up_backend_on_readiness_timeout(tmp_path):
    launcher = tmp_path / 'start.sh'
    launcher.write_text((Path(__file__).resolve().parents[1] / 'start.sh').read_text())
    pidfile = tmp_path / 'backend.pid'
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/bin/bash\nif [ "$1" = "-" ]; then\n cat >/dev/null\n echo "127.0.0.1 1"\n exit 0\nfi\necho $$ > "$TEST_PIDFILE"\nexec sleep 60\n')
    fake.chmod(0o755)
    curl = tmp_path / 'curl'
    curl.write_text('#!/bin/bash\nexit 1\n')
    curl.chmod(0o755)
    result = subprocess.run(['/bin/bash', str(launcher)],
                            env={**os.environ, 'PYTHON_BIN': str(fake), 'TEST_PIDFILE': str(pidfile),
                                 'PATH': str(tmp_path) + ':' + os.environ['PATH']},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert 'startup deadline' in result.stderr
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_chunked_body_limit_is_enforced():
    async def scenario():
        async def body():
            yield b'x' * 600000
            yield b'x' * 600000
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
            response = await client.post('/search', content=body())
            assert response.status_code == 413
    asyncio.run(scenario())


def test_rate_limit_and_liveness(monkeypatch):
    from backend.http import RequestBoundary
    monkeypatch.setenv('ACADEMICFORGE_REQUESTS_PER_MINUTE', '1')
    async def base(scope, receive, send):
        from starlette.responses import JSONResponse
        await JSONResponse({'status': 'ok'})(scope, receive, send)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=RequestBoundary(base)), base_url='http://test') as client:
            assert (await client.get('/config')).status_code == 200
            assert (await client.get('/config')).status_code == 429
            assert (await client.get('/live')).status_code == 200
    asyncio.run(scenario())


def test_source_retry_budget_respects_retry_after(monkeypatch):
    import backend.sources as sources
    response = Mock(status_code=429, headers={'Retry-After': '60'})
    call = Mock(return_value=response)
    monkeypatch.setattr(sources.requests, 'get', call)
    with pytest.raises(SourceUnavailable):
        sources.get_source('https://example.com')
    assert call.call_count == 1
    response.close.assert_called_once()


def test_arxiv_atom_response_is_normalized(monkeypatch):
    response = Mock(content=b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/2601.12345v1</id><title>Test paper</title><summary>Test abstract</summary>
      <author><name>Ada</name></author><published>2026-01-01T00:00:00Z</published>
      <link title="pdf" href="https://arxiv.org/pdf/2601.12345"/><category term="cs.AI"/>
      </entry></feed>''')
    monkeypatch.setattr(pipeline, 'get_source', lambda *args, **kwargs: response)
    papers, direct = pipeline.search_arxiv_candidates('2601.12345')
    assert direct is True
    assert papers[0]['authors'] == ['Ada']
    assert papers[0]['date'] == '2026-01-01'
    assert papers[0]['metadata']['categories'] == ['cs.AI']
    assert diagnostics()['sources']['arxiv'] == 'ok'


def test_streamlit_page_has_no_initial_render_errors(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(APIClient, 'get_config', lambda self: {
        'llm_provider': 'transformers',
        'llm_models': {'summary': 'test/local', 'guidance': 'test/local', 'research_plan': 'test/local'},
        'generation_modes': {'fast': {'model': 'test/local'}, 'deep': {'model': 'test/local'}},
    })
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py')).run(timeout=10)
    assert not page.exception


def test_ranking_degradation_is_visible_to_client(monkeypatch):
    paper = {**PAPER, 'metadata': {'dense_backend': 'lexical_fallback', 'reranker_backend': 'original_order'}}
    monkeypatch.setattr(api, 'retrieve_and_rank_papers', lambda *args: [paper])
    response = TestClient(api.app).post('/search', json={'query': 'test'})
    assert response.status_code == 200
    warnings = response.json()['warnings']
    assert any('keyword scoring' in warning for warning in warnings)
    assert any('reranker' in warning for warning in warnings)


def test_streamlit_search_generate_and_reuse_summary(monkeypatch):
    from streamlit.testing.v1 import AppTest
    calls = []
    config = {'llm_provider': 'transformers', 'llm_models': {'summary': 'test/local'},
              'generation_modes': {'fast': {'model': 'test/local'}, 'deep': {'model': 'test/local'}}}
    monkeypatch.setattr(APIClient, 'get_config', lambda self: config)
    def post(self, path, payload):
        calls.append(path)
        if path == '/search':
            return {'papers': [PAPER], 'warnings': []}
        return {'summary': 'A useful summary'}
    def stream(self, *args, **kwargs):
        self.last_generation = {}
        yield '# Research Focus\n'
        yield 'Build a useful system.'
    monkeypatch.setattr(APIClient, 'post_json', post)
    monkeypatch.setattr(APIClient, 'stream_post', stream)
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py')).run(timeout=10)
    page.text_input[0].input('test research')
    next(button for button in page.button if button.label == 'Search').click().run(timeout=10)
    assert not page.exception
    for _ in range(2):
        next(button for button in page.button if button.label == 'Generate Research Plan').click().run(timeout=10)
        assert not page.exception
        assert 'Build a useful system.' in page.session_state.research_plan
    assert calls.count('/summarize') == 1

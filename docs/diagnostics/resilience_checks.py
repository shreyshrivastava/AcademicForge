"""Reproduce review findings offline, without loading models or reading .env.

Historical baseline diagnostic for commit 73832e7. The fixed implementation is
covered by tests/test_resilience.py; this script intentionally targets the old
code and should be run only in a checkout of that baseline revision.

Run from that baseline repository root with its virtual environment:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python docs/diagnostics/resilience_checks.py

CONFIRMED means the reviewed defect is still reproducible, not that behavior is correct.
This diagnostic is evidence for the review; it is not a regression acceptance suite.
"""
import asyncio
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Import the app without allowing load_dotenv() to read developer credentials.
with patch('dotenv.load_dotenv', return_value=False):
    appmod = importlib.import_module('backend.app')
import backend.llm as llm
from backend.config import AppConfig
import frontend.api_client as clientmod
import httpx
import requests

BASE_ENV = {
    'LOCAL_LLM_PROVIDER': 'transformers',
    'LOCAL_LLM_MODEL': 'test/local-model',
    'ACADEMICFORGE_SEARCH_MODE': 'fast',
}


def record(name, observed):
    print(json.dumps({'check': name, 'status': 'CONFIRMED', 'observed': observed}))



def check_preload_entrypoint():
    env = {'PATH': os.defpath, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    result = subprocess.run([sys.executable, 'scripts/download_weights.py', '--help'],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "No module named 'backend'" in result.stderr
    record('Default startup preload entrypoint cannot import backend',
           {'exit_code': result.returncode, 'error': "ModuleNotFoundError: No module named 'backend'"})


def check_fireworks_opt_out():
    # Exercise exactly the model-selection part of the real startup script.
    startup = (ROOT / 'start.sh').read_text().splitlines()
    selection = '\n'.join(startup[12:32])
    command = selection + '\nexec "$DIAGNOSTIC_PYTHON" -c ' + "'from backend.config import get_config; c=get_config(); print(c.llm_research_plan_model)'"
    env = {**os.environ, **BASE_ENV, 'LOCAL_LLM_USE_FIREWORKS': 'false',
           'FIREWORKS_API_KEY': 'dummy-not-a-real-key', 'DIAGNOSTIC_PYTHON': sys.executable}
    for key in ('LOCAL_LLM_DEEP_MODEL', 'LOCAL_LLM_RESEARCH_PLAN_MODEL', 'REMOTE_LLM_USE_FOR_DEEP'):
        env.pop(key, None)
    model = subprocess.check_output(['/bin/bash', '-c', command], cwd=ROOT, env=env, text=True).strip()
    with patch.dict(os.environ, {**BASE_ENV, 'LOCAL_LLM_USE_FIREWORKS': 'false',
                               'LOCAL_LLM_RESEARCH_PLAN_MODEL': model}, clear=True), \
         patch.object(llm, '_generate_fireworks', return_value='cloud was selected') as cloud:
        result = llm.LLMService().generate('system', 'user', model=model)
        assert cloud.called
    record('Fireworks opt-out overridden by startup model selection', {'model': model, 'result': result})


def check_stream_fallback():
    def interrupted(*args, **kwargs):
        yield 'REMOTE PARTIAL. '
        raise RuntimeError('simulated upstream outage')
    env = {**BASE_ENV, 'REMOTE_LLM_USE_FOR_DEEP': 'true',
           'REMOTE_LLM_MODEL': 'test/remote-model', 'GROQ_API_KEY': 'dummy-not-a-real-key'}
    with patch.dict(os.environ, env, clear=True), \
         patch.object(llm, '_generate_remote_stream', interrupted), \
         patch.object(llm, '_generate_transformers', return_value='LOCAL FULL PLAN.'):
        result = ''.join(llm.LLMService().stream('system', 'user', model='test/remote-model'))
    assert result == 'REMOTE PARTIAL. LOCAL FULL PLAN.'
    record('Server fallback mixes partial remote output with a new local answer', result)


def check_client_replay():
    calls = []
    class Response:
        def raise_for_status(self):
            pass
        def iter_content(self, **kwargs):
            if len(calls) == 1:
                yield 'PARTIAL. '
                raise requests.Timeout('simulated read timeout')
            yield 'FULL PLAN.'
    def post(*args, **kwargs):
        calls.append(1)
        return Response()
    client = clientmod.APIClient.__new__(clientmod.APIClient)
    client.base_url = 'http://diagnostic.invalid'
    client.max_retries = 3
    client.timeout_generation = 240
    with patch.object(clientmod.requests, 'post', post), patch.object(clientmod.time, 'sleep'):
        result = ''.join(client.stream_post('/research-plan/stream', {}))
    assert result == 'PARTIAL. FULL PLAN.' and len(calls) == 2
    record('Frontend retries a stream after delivering partial text', {'requests': len(calls), 'output': result})


async def check_async_blocking():
    def slow_search(*args, **kwargs):
        time.sleep(0.2)
        return []
    async def ticker():
        started = time.perf_counter()
        await asyncio.sleep(0.01)
        return time.perf_counter() - started
    with patch.object(appmod, 'retrieve_and_rank_papers', slow_search):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=appmod.app), base_url='http://test') as client:
            delay, response = await asyncio.gather(ticker(), client.post('/search', json={'query': 'test'}))
    assert response.status_code == 200 and delay > 0.15
    record('Synchronous search blocks unrelated work on the event loop', {'10ms_timer_elapsed_ms': round(delay * 1000)})


async def check_readiness():
    fake_llm = Mock()
    fake_llm.cache_info.return_value = SimpleNamespace(currsize=1)
    with patch.dict(os.environ, BASE_ENV, clear=True), \
         patch.object(llm, '_load_transformers_model', fake_llm), \
         patch('backend.retrieval.dense._load_sentence_transformer', return_value=None), \
         patch('backend.retrieval.reranker.get_reranker_model', side_effect=RuntimeError('simulated reranker startup failure')):
        await appmod.startup_event()
        health = await appmod.health_check()
    assert health['models_ready'] is True and health['status'] == 'ok'
    record('Readiness hides failed retrieval warmup', {'models_ready': health['models_ready'], 'status': health['status']})


async def check_validation():
    with patch.object(appmod, 'retrieve_and_rank_papers', return_value=[]) as search, \
         patch.object(appmod, 'generate_ai_research_plan', return_value='mock plan') as generate:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=appmod.app), base_url='http://test') as client:
            empty_query = await client.post('/search', json={'query': '   '})
            empty_evidence = await client.post('/research-plan', json={'papers': [], 'generation_mode': 'invalid-mode'})
    assert empty_query.status_code == 200 and search.called
    assert empty_evidence.status_code == 200 and generate.called
    record('Invalid requests reach business logic', {'blank_query_status': empty_query.status_code, 'empty_papers_invalid_mode_status': empty_evidence.status_code})


def check_unhandled_jev_shape():
    import backend.jev as jev
    with patch.dict(os.environ, {**BASE_ENV, 'ACADEMICFORGE_SEARCH_MODE': 'quality',
                               'JEV_ENABLED': 'true', 'JEV_ENDPOINT': 'http://127.0.0.1:3000/v1/systemone'}, clear=True):
        config = AppConfig.from_env()
    response = Mock()
    response.json.return_value = []
    with patch.object(jev.requests, 'post', return_value=response):
        try:
            jev.enrich_papers_with_jev('unique malformed response diagnostic', [{'title': 'T', 'abstract': 'A'}], config)
        except AttributeError as exc:
            record('Valid JSON with wrong shape bypasses OpenJev fallback', type(exc).__name__)
        else:
            raise AssertionError('Expected the reviewed unhandled AttributeError')


def check_positive_remote_fallback():
    env = {**BASE_ENV, 'REMOTE_LLM_USE_FOR_DEEP': 'true',
           'REMOTE_LLM_MODEL': 'test/remote-model', 'GROQ_API_KEY': 'dummy-not-a-real-key'}
    with patch.dict(os.environ, env, clear=True), \
         patch.object(llm, '_generate_remote', side_effect=RuntimeError('simulated outage')), \
         patch.object(llm, '_generate_transformers', return_value='local answer'):
        result = llm.LLMService().generate('system', 'user', model='test/remote-model')
    assert result == 'local answer'
    print(json.dumps({'check': 'Non-streaming remote-to-local fallback exists', 'status': 'WORKING', 'observed': result}))


async def main():
    check_preload_entrypoint()
    check_fireworks_opt_out()
    check_stream_fallback()
    check_client_replay()
    await check_async_blocking()
    await check_readiness()
    await check_validation()
    check_unhandled_jev_shape()
    check_positive_remote_fallback()


if __name__ == '__main__':
    asyncio.run(main())

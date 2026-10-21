"""Regression cases for ambiguous acronyms and explicit search fields."""

from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

import backend.app as api
import backend.data_pipeline as pipeline
from backend.search_cache import cache_key
from backend.search_context import contextualize_query, matches_field
from backend.schemas import SearchRequest
from frontend.api_client import APIClient

AI = {
    'paper_id': 'ai', 'title': 'RAGAS: Automated Evaluation of Retrieval Augmented Generation',
    'abstract': 'Evaluation of retrieval and answer generation for large language models and LLM systems.',
    'authors': [], 'date': '2024', 'link': 'https://example.com/ai',
    'metadata': {'fields_of_study': ['Computer Science']},
}
MUSIC = {
    'paper_id': 'music', 'title': 'Ragas in Indian classical music',
    'abstract': 'Melodic structures and rhythm in traditional musical compositions.',
    'authors': [], 'date': '2024', 'link': 'https://example.com/music',
    'metadata': {'fields_of_study': ['Art']},
}
MUSIC_ML = {
    **MUSIC, 'paper_id': 'music-ml', 'title': 'Neural raga recognition',
    'abstract': 'A neural model for classifying ragas in Indian music using pitch features.',
    'metadata': {'fields_of_study': ['Computer Science']},
}


@pytest.mark.parametrize('query', ['RAGAS', 'rag as', 'Rag and AS'])
def test_computing_field_expands_ragas(query):
    expanded = contextualize_query(query, 'Computer science & AI')
    assert expanded == 'RAGAS retrieval augmented generation evaluation'


def test_rag_expansion_preserves_research_question():
    assert contextualize_query('RAG for medical diagnosis', 'AI') == 'retrieval augmented generation for medical diagnosis'


def test_rag_as_a_service_is_not_misread_as_ragas():
    assert contextualize_query('RAG as a service', 'AI') == 'retrieval augmented generation as a service'


def test_music_and_all_fields_do_not_force_computing_meaning():
    assert contextualize_query('ragas', '') == 'ragas'
    assert contextualize_query('ragas', 'All fields') == 'ragas'
    assert contextualize_query('ragas', 'Music & arts') == 'ragas music arts'


def test_computing_ragas_excludes_both_music_and_acoustic_ml_papers():
    assert matches_field(AI, 'rag as', 'Computer science & AI')
    assert not matches_field(MUSIC, 'rag as', 'Computer science & AI')
    assert not matches_field(MUSIC_ML, 'RAGAS', 'Computer science & AI')


def test_music_remains_searchable():
    assert matches_field(MUSIC, 'ragas', 'Music & arts')
    assert not matches_field(AI, 'ragas', 'Music & arts')


def test_custom_field_uses_title_abstract_and_metadata():
    paper = {'title': 'Entangled quantum systems', 'abstract': 'A study of quantum effects.',
             'metadata': {'fields_of_study': ['Physics']}}
    assert matches_field(paper, 'entanglement', 'quantum physics')
    assert not matches_field(MUSIC, 'entanglement', 'quantum physics')


def test_field_is_part_of_cache_identity():
    assert cache_key('ragas', [], 'fast', 'Music & arts') != cache_key('ragas', [], 'fast', 'Computer science & AI')
    assert cache_key('ragas', [], 'fast', 'AI') == cache_key('ragas', [], 'fast', ' ai ')


def test_api_passes_field_and_reports_interpreted_query(monkeypatch):
    retrieve = Mock(return_value=[AI])
    monkeypatch.setattr(api, 'retrieve_and_rank_papers', retrieve)
    response = TestClient(api.app).post('/search', json={'query': 'Rag and AS', 'domain': 'Computer science & AI'})
    assert response.status_code == 200
    retrieve.assert_called_once_with('Rag and AS', [], domain='Computer science & AI')
    assert response.json()['search_query'] == 'RAGAS retrieval augmented generation evaluation'
    assert response.json()['domain'] == 'Computer science & AI'


def test_api_limits_custom_field_length():
    response = TestClient(api.app).post('/search', json={'query': 'test', 'domain': 'a' * 121})
    assert response.status_code == 422


def test_domain_does_not_get_silently_relaxed(monkeypatch):
    monkeypatch.setattr(pipeline, 'search_live_candidates', lambda *args, **kwargs: ([MUSIC, MUSIC_ML], False))
    assert pipeline.retrieve_and_rank_papers('RAGAS', domain='Computer science & AI') == []


@pytest.mark.parametrize('domain,expected_ids', [
    ('Computer science & AI', ['ai-0', 'ai-1', 'ai-2']),
    ('Music & arts', ['music-0', 'music-1', 'music-2']),
])
def test_source_expansion_filtering_ranking_and_cache_honor_field(monkeypatch, domain, expected_ids):
    calls = []
    candidates = [{**AI, 'paper_id': f'ai-{i}'} for i in range(3)] + [
        {**MUSIC, 'paper_id': f'music-{i}'} for i in range(3)]
    def search(query, **kwargs):
        calls.append(query)
        return candidates, False
    rank_queries = []
    def rank(query, papers):
        rank_queries.append(query)
        return papers
    monkeypatch.setattr(pipeline, 'search_live_candidates', search)
    monkeypatch.setattr(pipeline, 'rank_papers', rank)
    monkeypatch.setattr(pipeline, 'select_evidence_set', lambda papers, **kwargs: papers)
    for _ in range(2):
        result = pipeline.retrieve_and_rank_papers('ragas', domain=domain)
        assert [paper['paper_id'] for paper in result] == expected_ids
    assert len(calls) == 2  # Initial + expanded once; second request is a cache hit.
    assert rank_queries == [contextualize_query('ragas', domain)]


def test_direct_paper_ids_are_not_rewritten(monkeypatch):
    captured = []
    def direct(query, **kwargs):
        captured.append(query)
        return [AI], True
    monkeypatch.setattr(pipeline, 'search_live_candidates', direct)
    assert pipeline.retrieve_and_rank_papers('2401.12345', domain='Computer science & AI')
    assert captured == ['2401.12345']


def test_streamlit_field_selection_reaches_backend_and_marks_stale_results(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    payloads = []
    monkeypatch.setattr(APIClient, 'get_config', lambda self: {
        'llm_models': {}, 'generation_modes': {'fast': {'model': 'local'}, 'deep': {'model': 'local'}},
    })
    def post(self, path, payload):
        payloads.append(payload)
        return {'papers': [AI], 'domain': payload['domain'],
                'search_query': contextualize_query(payload['query'], payload['domain'])}
    monkeypatch.setattr(APIClient, 'post_json', post)
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py')).run(timeout=10)
    page.text_input[0].input('rag as')
    page.selectbox(key='search_field').select('Computer science & AI').run(timeout=10)
    next(button for button in page.button if button.label == 'Search').click().run(timeout=10)
    assert not page.exception
    assert payloads[-1]['domain'] == 'Computer science & AI'
    assert page.session_state.last_search_query == 'RAGAS retrieval augmented generation evaluation'
    page.selectbox(key='search_field').select('Music & arts').run(timeout=10)
    assert any('Press Search' in item.value for item in page.info)
    assert page.session_state.last_search_domain == 'Computer science & AI'


def test_streamlit_custom_field_is_required_and_sent(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(APIClient, 'get_config', lambda self: {})
    post = Mock(return_value={'papers': []})
    monkeypatch.setattr(APIClient, 'post_json', post)
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'frontend/streamlit_app.py')).run(timeout=10)
    page.text_input[0].input('entanglement')
    page.selectbox(key='search_field').select('Other field…').run(timeout=10)
    next(button for button in page.button if button.label == 'Search').click().run(timeout=10)
    assert any('Enter a field' in item.value for item in page.warning)
    post.assert_not_called()
    page.text_input(key='custom_search_field').input('quantum physics')
    next(button for button in page.button if button.label == 'Search').click().run(timeout=10)
    assert not page.exception
    assert post.call_args.args[-1]['domain'] == 'quantum physics'


def test_research_plan_keeps_original_question_and_field(monkeypatch):
    generate = Mock(return_value='plan')
    monkeypatch.setattr(api, 'generate_ai_research_plan', generate)
    response = TestClient(api.app).post('/research-plan', json={
        'papers': [AI], 'query': 'Rag and AS for healthcare', 'domain': 'Computer science & AI',
    })
    assert response.status_code == 200
    goal = generate.call_args.args[2]
    assert 'Rag and AS for healthcare' in goal
    assert 'Search field: Computer science & AI' in goal
    assert 'RAGAS for healthcare retrieval augmented generation evaluation' in goal

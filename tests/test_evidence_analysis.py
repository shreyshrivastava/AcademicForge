"""Verify evidence transformations and keep recommendations distinct from facts."""

from frontend.evidence_analysis import analyze, condense, paper_guidance, paper_summary, research_plan

ABSTRACT = (
    "Research assistants handle increasing amounts of text. "
    "Many organizations maintain large document collections. "
    "We propose a retrieval augmented generation pipeline using BM25 and passage verification. "
    "We evaluate on SciFact using Recall@10 and nDCG@10. "
    "Our method improves Recall@10 by 12.5% over the baseline. "
    "The method fails on documents with noisy OCR. "
    "Our implementation is available for future research."
)
PAPER = {'title': 'Verified Retrieval for Research', 'abstract': ABSTRACT, 'url': 'https://example.org/retrieval'}


def test_summary_condenses_and_preserves_reported_measurements():
    condensed = condense(ABSTRACT, 'retrieval evaluation')
    assert len(condensed.split()) < len(ABSTRACT.split())
    info = analyze(PAPER)
    assert '12.5%' in info['signals']['result']
    assert 'SciFact' in info['datasets']
    assert 'Recall@10' in info['metrics']
    assert 'noisy OCR' in info['signals']['limitation']
    rendered = paper_summary(PAPER)
    assert ABSTRACT not in rendered
    assert 'Key concepts' in rendered
    assert 'https://example.org/retrieval' in rendered


def test_missing_evaluation_is_not_invented():
    paper = {'title': 'A conceptual essay', 'abstract': 'A discussion of scientific knowledge.'}
    info = analyze(paper)
    assert not info['metrics']
    assert not info['datasets']
    assert info['signals']['evaluation'] is None
    assert 'No named metric' in paper_guidance(paper)


def test_plan_is_specific_to_selected_evidence_and_task():
    retrieval = research_plan({'papers': [PAPER], 'query': 'retrieval verification'})
    forecast_paper = {'title': 'Solar Forecasting', 'abstract':
                      'We propose seasonal forecasting. We evaluate on hourly data using MAE. Forecasts improve MAE by 3%.'}
    forecast = research_plan({'papers': [forecast_paper], 'query': 'solar forecasting'})
    assert 'BM25 baseline' in retrieval
    assert 'SciFact' in retrieval
    assert 'Recall@10' in retrieval
    assert 'chronological' in forecast
    assert 'MAE' in forecast
    assert 'SciFact' not in forecast
    assert 'BM25 baseline' not in forecast
    assert 'Proposed starting point' in retrieval


def test_cross_paper_gaps_refer_to_missing_abstract_information():
    weak = {'title': 'Conceptual proposal', 'abstract': 'We introduce a new document representation.'}
    result = research_plan({'papers': [PAPER, weak], 'query': 'retrieval'})
    assert 'P2: the abstract does not explicitly describe evaluation procedure' in result
    assert 'P1 — Verified Retrieval' in result
    assert 'P2 — Conceptual proposal' in result
    assert 'No AI model' in result

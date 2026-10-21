"""Failure and admission contracts without downloading or loading model weights."""

from types import SimpleNamespace
import sys

import numpy as np

from frontend.small_model import SmallModel
from frontend.demo_runtime import DemoClient
from frontend import demo_runtime


def ready_model():
    model = SmallModel()
    model.state = "ready"
    model.encoder = SimpleNamespace(run=lambda *a: [np.zeros((1, 2, 4))])
    tokens = iter([4, 5, 1])
    def decode(*args):
        logits = np.zeros((1, 1, 8))
        logits[0, 0, next(tokens)] = 10
        return [logits]
    model.decoder = SimpleNamespace(run=decode, get_inputs=lambda: [
        SimpleNamespace(name=n) for n in ('input_ids', 'encoder_hidden_states', 'encoder_attention_mask')])
    model.tokenizer = SimpleNamespace(
        encode=lambda text: SimpleNamespace(ids=[2, 1], attention_mask=[1, 1]),
        decode=lambda *a, **k: "A concise research summary supported by the supplied evidence.",
    )
    return model


def test_disabled_model_does_not_start_loading():
    model = SmallModel()
    assert model.status() == "disabled"
    assert model.state == "not_started"
    assert model.generate("test") == (None, "model_disabled")


def test_missing_dependency_is_recoverable_without_exposing_details(monkeypatch):
    monkeypatch.setitem(sys.modules, 'onnxruntime', None)
    model = SmallModel()
    model._load()
    assert model.state == "unavailable"
    assert model.failure == "ModuleNotFoundError"
    model.retry()
    assert model.state == "not_started"


def test_generation_stops_at_eos_and_releases_capacity(monkeypatch):
    monkeypatch.setenv("ACADEMICFORGE_DEMO_MODEL_ENABLED", "true")
    model = ready_model()
    text, reason = model.generate("Summarize the evidence")
    assert "supported by" in text
    assert reason is None
    assert model.inference.acquire(blocking=False)
    model.inference.release()


def test_busy_or_timed_out_inference_uses_fallback(monkeypatch):
    monkeypatch.setenv("ACADEMICFORGE_DEMO_MODEL_ENABLED", "true")
    model = ready_model()
    model.inference.acquire()
    assert model.generate("test") == (None, "model_busy")
    model.inference.release()
    assert model.generate("test", seconds=-1) == (None, "model_timeout")


def test_generated_summary_has_actual_model_provenance(monkeypatch):
    monkeypatch.setattr(demo_runtime, 'get_model', lambda: SimpleNamespace(
        generate=lambda prompt: ("The proposed retrieval system checks evidence before generating its answer.", None)))
    result = DemoClient().post_json('/summarize', {'abstract': 'A retrieval system checks its evidence.'})
    assert 'Small-model summary' in result['summary']
    assert result['generation']['provider'] == 'onnx_cpu'
    assert not result['generation']['fallback']


def test_invented_numerical_result_is_rejected(monkeypatch):
    monkeypatch.setattr(demo_runtime, 'get_model', lambda: SimpleNamespace(
        generate=lambda prompt: ("The method achieved 99.9% accuracy in all the reported experiments.", None)))
    result = DemoClient().post_json('/summarize', {'abstract': 'We propose a retrieval method.'})
    assert '99.9' not in result['summary']
    assert result['generation']['fallback_reason'] == 'model_output_unusable'
    assert 'extractive' in result['generation']['fallback_message']


def test_model_overview_survives_plan_panel_rendering(monkeypatch):
    from frontend.components import split_into_sections, extract_bullets
    text = "The selected papers combine retrieval and evidence verification for generated answers."
    monkeypatch.setattr(demo_runtime, 'get_model', lambda: SimpleNamespace(generate=lambda prompt: (text, None)))
    result = DemoClient().post_json('/research-plan', {'query': 'retrieval', 'papers': [
        {'title': 'Retrieval study', 'abstract': 'We propose evidence verification.', 'url': 'https://example.org/paper'}]})
    findings = extract_bullets(split_into_sections(result['research_plan'])['Key Findings'])
    assert any(text in item for item in findings)
    assert result['generation']['provider'] == 'onnx_cpu'

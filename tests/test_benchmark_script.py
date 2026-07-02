import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.benchmark_llm as benchmark_llm


def test_preview_truncates_long_text():
    text = "word " * 400
    preview = benchmark_llm._preview(text, limit=80)
    assert len(preview) < len(text)
    assert preview.endswith("\n...")


def test_sample_papers_have_required_fields():
    for paper in benchmark_llm.SAMPLE_PAPERS:
        assert paper["title"]
        assert paper["abstract"]
        assert paper["link"]
        assert paper["date"]


def test_parse_models_accepts_repeated_and_comma_values():
    models = benchmark_llm._parse_models(["a,b", "c"])
    assert models == ["a", "b", "c"]


def test_quality_flags_catch_common_small_model_artifacts():
    flags = benchmark_llm.quality_flags("Okay, here's a result <end_of_turn> RRF")
    assert flags["chatty_preface"]
    assert flags["contains_end_of_turn"]
    assert flags["mentions_internal_retrieval"]


if __name__ == "__main__":
    test_preview_truncates_long_text()
    test_sample_papers_have_required_fields()
    test_parse_models_accepts_repeated_and_comma_values()
    test_quality_flags_catch_common_small_model_artifacts()
    print("benchmark script tests passed")

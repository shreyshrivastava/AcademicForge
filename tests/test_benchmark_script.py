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


if __name__ == "__main__":
    test_preview_truncates_long_text()
    test_sample_papers_have_required_fields()
    print("benchmark script tests passed")

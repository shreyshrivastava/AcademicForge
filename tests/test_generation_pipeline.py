import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.cache as cache
from backend.research_plan_generator import build_research_plan_context, generate_research_plan
from backend.research_plan_generator import generate_paper_guidance
from backend.research_plan_generator import paper_guidance_cache_key
from backend.research_plan_generator import paper_guidance_cache_status
from backend.research_plan_generator import research_plan_cache_status
from backend.research_plan_generator import stream_research_plan
import backend.research_plan_generator as research_plan_generator
import backend.summarizer as summarizer


PAPER = {
    "paper_id": "paper-1",
    "title": "Reducing Hallucination in Retrieval-Augmented Generation",
    "abstract": " ".join(["long-abstract"] * 300),
    "authors": ["Ada", "Bert", "Cy", "Dee", "Eve"],
    "source": "test",
    "url": "https://example.com/paper-1",
    "link": "https://example.com/paper-1",
    "date": "2026-01-01",
    "bm25_rank": 1,
    "dense_rank": 2,
    "rrf_score": 0.031,
}

SUMMARY = """
Core idea
Detect hallucinations in RAG answers by comparing generated claims with retrieved evidence.
Method
Use retrieval evidence, factuality signals, and lightweight verification features.
Why it matters
It gives builders a practical way to reduce unsupported answers.
Implementation notes
Start with claim extraction, evidence matching, and a calibrated hallucination score.
Limitations or unknowns
Domain transfer and ambiguous evidence remain open problems.
""".strip()


def test_research_plan_context_is_compact_and_structured():
    context = build_research_plan_context([PAPER], [SUMMARY])
    assert "[Evidence 1] Reducing Hallucination" in context
    assert "Retrieval:" not in context
    assert "RRF" not in context
    assert "Prior extracted notes:" in context
    assert " ".join(["long-abstract"] * 80) not in context
    assert len(context) < 1800


def test_generate_research_plan_uses_compact_context():
    captured = {}
    original_generate_text = research_plan_generator.generate_text
    original_cache_dir = cache.CACHE_DIR
    research_plan_generator.RESEARCH_PLAN_CACHE.clear()

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["token_budget"] = token_budget
        captured["task"] = task
        return "research plan"

    research_plan_generator.generate_text = fake_generate_text
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        result = generate_research_plan([PAPER], [SUMMARY], query="build a RAG hallucination detector")
        cache.CACHE_DIR = original_cache_dir
        research_plan_generator.generate_text = original_generate_text
        research_plan_generator.RESEARCH_PLAN_CACHE.clear()

    assert result == "research plan"
    assert captured["token_budget"] == 1900
    assert captured["task"] == "research_plan"
    assert "AI Research Engineer" in captured["system_prompt"]
    assert "User Goal Analysis" in captured["user_prompt"]
    assert "Research Focus" in captured["user_prompt"]
    assert "Key Findings" in captured["user_prompt"]
    assert "Research Gaps" in captured["user_prompt"]
    assert "Recommended Build" in captured["user_prompt"]
    assert "Do not output Evidence Used" in captured["user_prompt"]
    assert "Paper Relevance Analysis" not in captured["user_prompt"]
    assert "Implementation Roadmap" not in captured["user_prompt"]
    assert "build a RAG hallucination detector" in captured["user_prompt"]
    assert "1. What this paper is about" not in captured["user_prompt"]
    assert "4. Step-by-step reading plan" not in captured["user_prompt"]
    assert " ".join(["long-abstract"] * 80) not in captured["user_prompt"]


def test_generate_paper_guidance_uses_paper_specific_prompt():
    captured = {}
    original_generate_text = research_plan_generator.generate_text
    original_cache_dir = cache.CACHE_DIR
    research_plan_generator.PAPER_GUIDANCE_CACHE.clear()

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["token_budget"] = token_budget
        captured["task"] = task
        return "paper guidance"

    research_plan_generator.generate_text = fake_generate_text
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        result = generate_paper_guidance(PAPER)
        cache.CACHE_DIR = original_cache_dir
        research_plan_generator.generate_text = original_generate_text
        research_plan_generator.PAPER_GUIDANCE_CACHE.clear()

    assert result == "paper guidance"
    assert captured["token_budget"] == 1100
    assert captured["task"] == "research_plan"
    assert "Paragraph 1: why this paper matters" in captured["user_prompt"]
    assert "Paragraph 2: how to use it in a project" in captured["user_prompt"]
    assert "Paragraph 3: what to watch out for or verify" in captured["user_prompt"]
    assert "Builder Goal Fit" not in captured["user_prompt"]
    assert "1. What this paper is about" not in captured["user_prompt"]
    assert "4. Step-by-step reading plan" not in captured["user_prompt"]
    assert "8. Difficulty level" not in captured["user_prompt"]


def test_paper_guidance_cache_reuses_disk_cache_and_invalidates_on_content_change():
    calls = {"count": 0}
    original_generate_text = research_plan_generator.generate_text
    original_cache_dir = cache.CACHE_DIR
    research_plan_generator.PAPER_GUIDANCE_CACHE.clear()

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        calls["count"] += 1
        return "disk cached paper guidance"

    changed_paper = dict(PAPER)
    changed_paper["abstract"] = "A changed abstract should produce a different cache key."

    research_plan_generator.generate_text = fake_generate_text
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        first = generate_paper_guidance(PAPER)
        research_plan_generator.PAPER_GUIDANCE_CACHE.clear()
        second = generate_paper_guidance(PAPER)
        assert first == "disk cached paper guidance"
        assert second == "disk cached paper guidance"
        assert calls["count"] == 1
        assert paper_guidance_cache_status(PAPER) == {
            "cached": True,
            "cache": "memory",
        }
        assert paper_guidance_cache_key(PAPER) != paper_guidance_cache_key(changed_paper)
        cache.CACHE_DIR = original_cache_dir
        research_plan_generator.generate_text = original_generate_text
        research_plan_generator.PAPER_GUIDANCE_CACHE.clear()


def test_summary_cache_reuses_existing_summary():
    calls = {"count": 0}
    original_generate_text = summarizer.generate_text
    original_cache_dir = cache.CACHE_DIR
    summarizer.SUMMARY_CACHE.clear()

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        calls["count"] += 1
        return "cached summary"

    summarizer.generate_text = fake_generate_text
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        first = summarizer.summarize_paper(PAPER)
        second = summarizer.summarize_paper(PAPER)
        cache.CACHE_DIR = original_cache_dir
        summarizer.generate_text = original_generate_text
        summarizer.SUMMARY_CACHE.clear()

    assert first == "cached summary"
    assert second == "cached summary"
    assert calls["count"] == 1


def test_summary_disk_cache_reuses_existing_summary_after_memory_clear():
    calls = {"count": 0}
    original_generate_text = summarizer.generate_text
    original_cache_dir = cache.CACHE_DIR
    summarizer.SUMMARY_CACHE.clear()

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        calls["count"] += 1
        return "disk cached summary"

    summarizer.generate_text = fake_generate_text
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        first = summarizer.summarize_paper(PAPER)
        summarizer.SUMMARY_CACHE.clear()
        second = summarizer.summarize_paper(PAPER)
        cache.CACHE_DIR = original_cache_dir
        summarizer.generate_text = original_generate_text
        summarizer.SUMMARY_CACHE.clear()

    assert first == "disk cached summary"
    assert second == "disk cached summary"
    assert calls["count"] == 1


def test_research_plan_disk_cache_reuses_existing_plan_after_memory_clear():
    calls = {"count": 0}
    original_generate_text = research_plan_generator.generate_text
    original_cache_dir = cache.CACHE_DIR
    research_plan_generator.RESEARCH_PLAN_CACHE.clear()

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        calls["count"] += 1
        return "disk cached research plan"

    research_plan_generator.generate_text = fake_generate_text
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        first = generate_research_plan([PAPER], [SUMMARY], query="build detector")
        research_plan_generator.RESEARCH_PLAN_CACHE.clear()
        second = generate_research_plan([PAPER], [SUMMARY], query="build detector")
        cache.CACHE_DIR = original_cache_dir
        research_plan_generator.generate_text = original_generate_text
        research_plan_generator.RESEARCH_PLAN_CACHE.clear()

    assert first == "disk cached research plan"
    assert second == "disk cached research plan"
    assert calls["count"] == 1


def test_research_plan_cache_status_reports_miss_and_disk_hit():
    original_generate_text = research_plan_generator.generate_text
    original_cache_dir = cache.CACHE_DIR
    research_plan_generator.RESEARCH_PLAN_CACHE.clear()

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        return "status research plan"

    research_plan_generator.generate_text = fake_generate_text
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        assert research_plan_cache_status([PAPER], [SUMMARY], query="build detector") == {
            "cached": False,
            "cache": "miss",
        }
        generate_research_plan([PAPER], [SUMMARY], query="build detector")
        research_plan_generator.RESEARCH_PLAN_CACHE.clear()
        assert research_plan_cache_status([PAPER], [SUMMARY], query="build detector") == {
            "cached": True,
            "cache": "disk",
        }
        cache.CACHE_DIR = original_cache_dir
        research_plan_generator.generate_text = original_generate_text
        research_plan_generator.RESEARCH_PLAN_CACHE.clear()


def test_stream_research_plan_yields_chunks_and_caches_result():
    calls = {"count": 0}
    original_stream = research_plan_generator.generate_text_stream
    original_cache_dir = cache.CACHE_DIR
    research_plan_generator.RESEARCH_PLAN_CACHE.clear()

    def fake_stream(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        calls["count"] += 1
        yield "```markdown\n"
        yield "streamed "
        yield "research plan"
        yield "\n```"

    research_plan_generator.generate_text_stream = fake_stream
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        first = "".join(stream_research_plan([PAPER], [SUMMARY], query="build detector"))
        research_plan_generator.RESEARCH_PLAN_CACHE.clear()
        second = "".join(stream_research_plan([PAPER], [SUMMARY], query="build detector"))
        cache.CACHE_DIR = original_cache_dir
        research_plan_generator.generate_text_stream = original_stream
        research_plan_generator.RESEARCH_PLAN_CACHE.clear()

    assert first == "streamed research plan"
    assert second == "streamed research plan"
    assert calls["count"] == 1


if __name__ == "__main__":
    test_research_plan_context_is_compact_and_structured()
    test_generate_research_plan_uses_compact_context()
    test_generate_paper_guidance_uses_paper_specific_prompt()
    test_paper_guidance_cache_reuses_disk_cache_and_invalidates_on_content_change()
    test_summary_cache_reuses_existing_summary()
    test_summary_disk_cache_reuses_existing_summary_after_memory_clear()
    test_research_plan_disk_cache_reuses_existing_plan_after_memory_clear()
    test_research_plan_cache_status_reports_miss_and_disk_hit()
    test_stream_research_plan_yields_chunks_and_caches_result()
    print("generation pipeline tests passed")

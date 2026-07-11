import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.research_plan_generator import build_research_plan_context, generate_research_plan
from backend.research_plan_generator import stream_research_plan
import backend.research_plan_generator as research_plan_generator
from backend.guidance_generator import generate_paper_guidance
import backend.guidance_generator as guidance_generator
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

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["token_budget"] = token_budget
        captured["task"] = task
        return "research plan"

    research_plan_generator.generate_text = fake_generate_text
    try:
        result = generate_research_plan([PAPER], [SUMMARY], query="build a RAG hallucination detector")
    finally:
        research_plan_generator.generate_text = original_generate_text

    assert result == "research plan"
    assert captured["token_budget"] == 1900
    assert captured["task"] == "research_plan"
    assert "AI Research Engineer" in captured["system_prompt"]
    assert "# User Goal Analysis" not in captured["user_prompt"]
    assert "Research Focus" in captured["user_prompt"]
    assert "Key Findings" in captured["user_prompt"]
    assert "Research Gaps" in captured["user_prompt"]
    assert "Recommended Build" in captured["user_prompt"]
    assert "Evaluation Strategy" in captured["user_prompt"]
    assert "Paper Relevance Analysis" not in captured["user_prompt"]
    assert "Implementation Roadmap" not in captured["user_prompt"]
    assert "build a RAG hallucination detector" in captured["user_prompt"]
    assert "1. What this paper is about" not in captured["user_prompt"]
    assert "4. Step-by-step reading plan" not in captured["user_prompt"]
    assert " ".join(["long-abstract"] * 80) not in captured["user_prompt"]


def test_generate_paper_guidance_uses_paper_specific_prompt():
    captured = {}
    original_generate_text = guidance_generator.generate_text

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["token_budget"] = token_budget
        captured["task"] = task
        return "paper guidance"

    guidance_generator.generate_text = fake_generate_text
    try:
        result = generate_paper_guidance(PAPER)
    finally:
        guidance_generator.generate_text = original_generate_text

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


if __name__ == "__main__":
    test_research_plan_context_is_compact_and_structured()
    test_generate_research_plan_uses_compact_context()
    test_generate_paper_guidance_uses_paper_specific_prompt()
    print("generation pipeline tests passed")

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.config as config
import backend.llm as llm
import backend.research_plan_generator as research_plan_generator
import backend.summarizer as summarizer


def test_summary_uses_summary_task():
    captured = {}
    original_generate_text = summarizer.generate_text

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        captured["task"] = task
        return "summary"

    summarizer.generate_text = fake_generate_text
    try:
        summarizer.summarize_paper(
            {
                "paper_id": "x",
                "title": "Paper",
                "authors": [],
                "abstract": "Abstract",
            }
        )
    finally:
        summarizer.generate_text = original_generate_text

    assert captured["task"] == "summary"


def test_research_plan_uses_research_plan_task():
    captured = {}
    original_generate_text = research_plan_generator.generate_text

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        captured["task"] = task
        return "research plan"

    research_plan_generator.generate_text = fake_generate_text
    try:
        research_plan_generator.generate_research_plan(
            [
                {
                    "title": "Paper",
                    "authors": [],
                    "abstract": "Abstract",
                    "date": "2026-01-01",
                    "link": "https://example.com",
                }
            ],
            ["Core idea\nIdea"],
        )
    finally:
        research_plan_generator.generate_text = original_generate_text

    assert captured["task"] == "research_plan"


def test_clean_response_preamble_removal():
    from backend.llm import _clean_response
    dirty_output = (
        "Summary\n"
        "We are asked: \"Write a concise summary in 2-4 sentences.\" The summary should answer: what does this paper say? Be specific, avoid hype.\n"
        "The paper title: \"Expansive Participatory AI: Supporting Dreaming within Inequitable Institutions\"\n"
        "Authors: Michael Alan Chang, Shiran Dudy\n"
        "Abstract: \"Participatory Artificial Intelligence (PAI) has recently gained interest...\"\n\n"
        "So the paper proposes co-design principles... The summary should capture that.\n\n"
        "I'll craft a concise summary: The paper argues that institutional power dynamics limit "
        "the transformative potential of Participatory AI, and it proposes co-design principles "
        "aimed at enabling youth stakeholders to realize expansive aspirations within inequitable institutions.\n"
        "(That's one sentence, but I can expand slightly.) I need 2-4 sentences. I'll mention that..."
    )
    cleaned = _clean_response(dirty_output)
    expected = (
        "The paper argues that institutional power dynamics limit the transformative potential "
        "of Participatory AI, and it proposes co-design principles aimed at enabling youth "
        "stakeholders to realize expansive aspirations within inequitable institutions."
    )
    assert cleaned == expected


def _restore_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    test_summary_uses_summary_task()
    test_research_plan_uses_research_plan_task()
    test_clean_response_preamble_removal()
    print("llm routing tests passed")

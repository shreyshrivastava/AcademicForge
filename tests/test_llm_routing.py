import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.cache as cache
import backend.llm as llm
import backend.roadmap_generator as roadmap_generator
import backend.summarizer as summarizer


def test_task_model_config_uses_task_overrides():
    old_default = os.environ.get("LOCAL_LLM_MODEL")
    old_summary = os.environ.get("LOCAL_LLM_SUMMARY_MODEL")
    old_roadmap = os.environ.get("LOCAL_LLM_ROADMAP_MODEL")

    os.environ["LOCAL_LLM_MODEL"] = "default-model"
    os.environ["LOCAL_LLM_SUMMARY_MODEL"] = "summary-model"
    os.environ["LOCAL_LLM_ROADMAP_MODEL"] = "roadmap-model"
    try:
        config = llm.task_model_config()
    finally:
        _restore_env("LOCAL_LLM_MODEL", old_default)
        _restore_env("LOCAL_LLM_SUMMARY_MODEL", old_summary)
        _restore_env("LOCAL_LLM_ROADMAP_MODEL", old_roadmap)

    assert config == {
        "default": "default-model",
        "summary": "summary-model",
        "roadmap": "roadmap-model",
    }


def test_summary_uses_summary_task():
    captured = {}
    original_generate_text = summarizer.generate_text
    original_cache_dir = cache.CACHE_DIR
    summarizer.SUMMARY_CACHE.clear()

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        captured["task"] = task
        return "summary"

    summarizer.generate_text = fake_generate_text
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        summarizer.summarize_paper(
            {
                "paper_id": "x",
                "title": "Paper",
                "authors": [],
                "abstract": "Abstract",
            }
        )
        cache.CACHE_DIR = original_cache_dir
        summarizer.generate_text = original_generate_text
        summarizer.SUMMARY_CACHE.clear()

    assert captured["task"] == "summary"


def test_roadmap_uses_roadmap_task():
    captured = {}
    original_generate_text = roadmap_generator.generate_text
    original_cache_dir = cache.CACHE_DIR
    roadmap_generator.ROADMAP_CACHE.clear()

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        captured["task"] = task
        return "roadmap"

    roadmap_generator.generate_text = fake_generate_text
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        roadmap_generator.generate_roadmap(
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
        cache.CACHE_DIR = original_cache_dir
        roadmap_generator.generate_text = original_generate_text
        roadmap_generator.ROADMAP_CACHE.clear()

    assert captured["task"] == "roadmap"


def _restore_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    test_task_model_config_uses_task_overrides()
    test_summary_uses_summary_task()
    test_roadmap_uses_roadmap_task()
    print("llm routing tests passed")

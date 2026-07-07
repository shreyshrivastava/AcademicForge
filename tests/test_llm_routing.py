import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.cache as cache
import backend.config as config
import backend.llm as llm
import backend.research_plan_generator as research_plan_generator
import backend.summarizer as summarizer


def test_provider_aliases_normalize_gpu_backends():
    old_provider = os.environ.get("LOCAL_LLM_PROVIDER")
    old_backend = os.environ.get("LLM_BACKEND")
    os.environ["LOCAL_LLM_PROVIDER"] = "roc"
    os.environ.pop("LLM_BACKEND", None)
    try:
        assert config.get_config().llm_provider == "transformers"
        assert llm.provider_name() == "transformers"
    finally:
        _restore_env("LOCAL_LLM_PROVIDER", old_provider)
        _restore_env("LLM_BACKEND", old_backend)


def test_task_model_config_uses_task_overrides():
    old_default = os.environ.get("LOCAL_LLM_MODEL")
    old_summary = os.environ.get("LOCAL_LLM_SUMMARY_MODEL")
    old_research_plan = os.environ.get("LOCAL_LLM_RESEARCH_PLAN_MODEL")

    os.environ["LOCAL_LLM_MODEL"] = "default-model"
    os.environ["LOCAL_LLM_SUMMARY_MODEL"] = "summary-model"
    os.environ["LOCAL_LLM_RESEARCH_PLAN_MODEL"] = "research-plan-model"
    try:
        config = llm.task_model_config()
    finally:
        _restore_env("LOCAL_LLM_MODEL", old_default)
        _restore_env("LOCAL_LLM_SUMMARY_MODEL", old_summary)
        _restore_env("LOCAL_LLM_RESEARCH_PLAN_MODEL", old_research_plan)

    assert config == {
        "default": "default-model",
        "summary": "summary-model",
        "research_plan": "research-plan-model",
        "roadmap": "research-plan-model",
    }


def test_task_model_config_accepts_legacy_roadmap_override():
    old_default = os.environ.get("LOCAL_LLM_MODEL")
    old_research_plan = os.environ.get("LOCAL_LLM_RESEARCH_PLAN_MODEL")
    old_roadmap = os.environ.get("LOCAL_LLM_ROADMAP_MODEL")

    os.environ["LOCAL_LLM_MODEL"] = "default-model"
    os.environ.pop("LOCAL_LLM_RESEARCH_PLAN_MODEL", None)
    os.environ["LOCAL_LLM_ROADMAP_MODEL"] = "legacy-roadmap-model"
    try:
        model_config = llm.task_model_config()
    finally:
        _restore_env("LOCAL_LLM_MODEL", old_default)
        _restore_env("LOCAL_LLM_RESEARCH_PLAN_MODEL", old_research_plan)
        _restore_env("LOCAL_LLM_ROADMAP_MODEL", old_roadmap)

    assert model_config["research_plan"] == "legacy-roadmap-model"
    assert model_config["roadmap"] == "legacy-roadmap-model"


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


def test_research_plan_uses_research_plan_task():
    captured = {}
    original_generate_text = research_plan_generator.generate_text
    original_cache_dir = cache.CACHE_DIR
    research_plan_generator.RESEARCH_PLAN_CACHE.clear()

    def fake_generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
        captured["task"] = task
        return "research plan"

    research_plan_generator.generate_text = fake_generate_text
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
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
        cache.CACHE_DIR = original_cache_dir
        research_plan_generator.generate_text = original_generate_text
        research_plan_generator.RESEARCH_PLAN_CACHE.clear()

    assert captured["task"] == "research_plan"


def _restore_env(name, value):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    test_provider_aliases_normalize_gpu_backends()
    test_task_model_config_uses_task_overrides()
    test_task_model_config_accepts_legacy_roadmap_override()
    test_summary_uses_summary_task()
    test_research_plan_uses_research_plan_task()
    print("llm routing tests passed")

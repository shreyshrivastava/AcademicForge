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


def test_qwen_provider_and_custom_base_url():
    import requests
    from unittest.mock import patch

    old_provider = os.environ.get("LOCAL_LLM_PROVIDER")
    old_openai_base = os.environ.get("OPENAI_BASE_URL")
    old_qwen_base = os.environ.get("QWEN_BASE_URL")

    os.environ["LOCAL_LLM_PROVIDER"] = "qwen"
    os.environ.pop("OPENAI_BASE_URL", None)
    os.environ.pop("QWEN_BASE_URL", None)

    try:
        cfg = config.AppConfig.from_env()
        assert cfg.llm_provider == "qwen"
        assert cfg.llm_model == "qwen-plus"

        # Test custom base URL override for qwen
        with patch("backend.llm.QWEN_BASE_URL", "https://custom-qwen.com/v1"), \
             patch("backend.llm.QWEN_API_KEY", "dummy-key"), \
             patch("requests.post") as mock_post:
            
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "qwen reply"}}]
            }
            service = llm.LLMService(provider="qwen")
            res = service.generate("system", "user", model="qwen-plus")
            assert res == "qwen reply"
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == "https://custom-qwen.com/v1/chat/completions"
            assert kwargs["headers"]["Authorization"] == "Bearer dummy-key"

        # Test custom base URL override for openai
        os.environ["LOCAL_LLM_PROVIDER"] = "openai"

        with patch("backend.llm.OPENAI_BASE_URL", "https://custom-openai.com/v2"), \
             patch("backend.llm.OPENAI_API_KEY", "dummy-openai-key"), \
             patch("requests.post") as mock_post:

            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": "openai reply"}}]
            }
            service = llm.LLMService(provider="openai")
            res = service.generate("system", "user")
            assert res == "openai reply"
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == "https://custom-openai.com/v2/chat/completions"
            assert kwargs["headers"]["Authorization"] == "Bearer dummy-openai-key"

    finally:
        _restore_env("LOCAL_LLM_PROVIDER", old_provider)
        _restore_env("OPENAI_BASE_URL", old_openai_base)
        _restore_env("QWEN_BASE_URL", old_qwen_base)


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
    test_provider_aliases_normalize_gpu_backends()
    test_task_model_config_uses_task_overrides()
    test_summary_uses_summary_task()
    test_research_plan_uses_research_plan_task()
    test_qwen_provider_and_custom_base_url()
    test_clean_response_preamble_removal()
    print("llm routing tests passed")

"""OpenAI-compatible remote generation: Groq-style providers and Fireworks."""
import os
import logging

from backend.config import env_number
from backend.llm_common import LocalLLMError, max_tokens

logger = logging.getLogger(__name__)


def _get_openai_client():
    try:
        from openai import OpenAI
        return OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ.get("FIREWORKS_API_KEY"),
            timeout=env_number("ACADEMICFORGE_CLOUD_TIMEOUT_SECONDS", 45, float, 1, 120),
            max_retries=0
        )
    except ImportError as exc:
        raise LocalLLMError(
            "OpenAI client not installed. Run `pip install openai`."
        ) from exc


def _get_remote_client(base_url, api_key):
    try:
        from openai import OpenAI
        return OpenAI(base_url=base_url, api_key=api_key,
                      timeout=env_number("ACADEMICFORGE_CLOUD_TIMEOUT_SECONDS", 45, float, 1, 120), max_retries=0)
    except ImportError as exc:
        raise LocalLLMError(
            "OpenAI client not installed. Run `pip install openai`."
        ) from exc


def _remote_request_options(token_budget):
    base_budget = token_budget or max_tokens()
    return {
        "max_tokens": min(base_budget * 2 + 500, 4096),
        "temperature": 0.6,
    }


def _generate_remote(system_prompt, user_prompt, token_budget, selected_model, base_url, api_key, provider):
    client = _get_remote_client(base_url, api_key)
    try:
        options = _remote_request_options(token_budget)
        if provider == "groq":
            options["reasoning_effort"] = os.getenv("REMOTE_LLM_REASONING_EFFORT", "medium")
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **options,
            stream=False,
        )
        return response.choices[0].message.content
    finally:
        client.close()


def _generate_remote_stream(system_prompt, user_prompt, token_budget, selected_model, base_url, api_key, provider):
    client = _get_remote_client(base_url, api_key)
    try:
        options = _remote_request_options(token_budget)
        if provider == "groq":
            options["reasoning_effort"] = os.getenv("REMOTE_LLM_REASONING_EFFORT", "medium")
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **options,
            stream=True,
        )
        for chunk in response:
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content is not None:
                yield content
    finally:
        client.close()


def _generate_fireworks(system_prompt, user_prompt, token_budget, selected_model):
    client = _get_openai_client()
    try:
        # Tune parameters specifically for DeepSeek via Fireworks
        # We dynamically cap max_tokens to save credits while leaving enough room to think
        base_budget = token_budget or max_tokens()
        fw_max_tokens = min(base_budget * 2 + 500, 2048)
        fw_temperature = 0.6

        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=fw_max_tokens,
            temperature=fw_temperature,
            stream=False,
        )
        return response.choices[0].message.content
    finally:
        client.close()


def _generate_fireworks_stream(system_prompt, user_prompt, token_budget, selected_model):
    client = _get_openai_client()
    try:
        # Tune parameters specifically for DeepSeek via Fireworks
        # We dynamically cap max_tokens to save credits while leaving enough room to think
        base_budget = token_budget or max_tokens()
        fw_max_tokens = min(base_budget * 2 + 500, 2048)
        fw_temperature = 0.6

        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=fw_max_tokens,
            temperature=fw_temperature,
            stream=True,
        )
        for chunk in response:
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content is not None:
                yield content
    finally:
        client.close()

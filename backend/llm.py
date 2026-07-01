import os
import re
from functools import lru_cache


SUPPORTED_PROVIDER = "mlx"
DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"
DEFAULT_MAX_TOKENS = 700
DEFAULT_TEMPERATURE = 0.2
MODEL_ENV = "LOCAL_LLM_MODEL"
MAX_TOKENS_ENV = "LOCAL_LLM_MAX_TOKENS"
TEMPERATURE_ENV = "LOCAL_LLM_TEMPERATURE"


class LocalLLMError(RuntimeError):
    pass


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def provider_name():
    return os.getenv("LOCAL_LLM_PROVIDER", SUPPORTED_PROVIDER).strip().lower()


def model_name(task=None):
    if task:
        task_model = os.getenv(f"LOCAL_LLM_{task.upper()}_MODEL")
        if task_model:
            return task_model.strip()
    return os.getenv(MODEL_ENV, DEFAULT_MODEL).strip()


def task_model_config():
    return {
        "default": model_name(),
        "summary": model_name("summary"),
        "roadmap": model_name("roadmap"),
    }


def max_tokens(default=DEFAULT_MAX_TOKENS):
    return _env_int(MAX_TOKENS_ENV, default)


def temperature():
    return _env_float(TEMPERATURE_ENV, DEFAULT_TEMPERATURE)


@lru_cache(maxsize=4)
def _load_mlx_model(selected_model):
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise LocalLLMError(
            "MLX runtime is not installed. Install it with: pip install mlx-lm"
        ) from exc

    return load(selected_model)


def _chat_prompt(tokenizer, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            pass

    return (
        f"System:\n{system_prompt}\n\n"
        f"User:\n{user_prompt}\n\n"
        "Assistant:\n"
    )


def generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
    provider = provider_name()
    selected_model = model or model_name(task)
    if provider == SUPPORTED_PROVIDER:
        return _clean_response(
            _generate_mlx(system_prompt, user_prompt, token_budget, selected_model)
        )

    # Future CUDA/ROCm support should be added as a separate backend path here.
    raise LocalLLMError(
        f"Unknown LOCAL_LLM_PROVIDER={provider!r}. "
        f"AcademicForge currently supports {SUPPORTED_PROVIDER!r}."
    )


def generate_text_stream(system_prompt, user_prompt, token_budget=None, task=None, model=None):
    provider = provider_name()
    selected_model = model or model_name(task)
    if provider == SUPPORTED_PROVIDER:
        yield from _generate_mlx_stream(system_prompt, user_prompt, token_budget, selected_model)
        return

    # Future CUDA/ROCm streaming should be added as a separate backend path here.
    raise LocalLLMError(
        f"Unknown LOCAL_LLM_PROVIDER={provider!r}. "
        f"AcademicForge currently supports {SUPPORTED_PROVIDER!r}."
    )


def _clean_response(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    fence_match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return text.strip()


def _generate_mlx(system_prompt, user_prompt, token_budget, selected_model):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load_mlx_model(selected_model)
    prompt = _chat_prompt(tokenizer, system_prompt, user_prompt)
    sampler = make_sampler(temp=temperature())
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        verbose=False,
        max_tokens=token_budget or max_tokens(),
        sampler=sampler,
    )
    return text.strip()


def _generate_mlx_stream(system_prompt, user_prompt, token_budget, selected_model):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load_mlx_model(selected_model)
    prompt = _chat_prompt(tokenizer, system_prompt, user_prompt)
    sampler = make_sampler(temp=temperature())
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=token_budget or max_tokens(),
        sampler=sampler,
    ):
        if response.text:
            yield response.text

import os
import re
from functools import lru_cache


DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"


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
    return os.getenv("LOCAL_LLM_PROVIDER", "mlx").strip().lower()


def model_name(task=None):
    if task:
        task_model = os.getenv(f"LOCAL_LLM_{task.upper()}_MODEL")
        if task_model:
            return task_model.strip()
    return os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL).strip()


def task_model_config():
    return {
        "default": model_name(),
        "summary": model_name("summary"),
        "roadmap": model_name("roadmap"),
    }


def max_tokens(default=700):
    return _env_int("LOCAL_LLM_MAX_TOKENS", default)


def temperature():
    return _env_float("LOCAL_LLM_TEMPERATURE", 0.2)


@lru_cache(maxsize=4)
def _load_mlx_model(selected_model):
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise LocalLLMError(
            "MLX runtime is not installed. Install it with: pip install mlx-lm"
        ) from exc

    return load(selected_model)


@lru_cache(maxsize=4)
def _load_transformers_model(selected_model):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise LocalLLMError(
            "Transformers runtime is not installed. Install transformers and torch."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    model = AutoModelForCausalLM.from_pretrained(
        selected_model,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer, torch


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
    if provider == "mlx":
        return _clean_response(
            _generate_mlx(system_prompt, user_prompt, token_budget, selected_model)
        )
    if provider in {"transformers", "torch", "rocm"}:
        return _clean_response(
            _generate_transformers(system_prompt, user_prompt, token_budget, selected_model)
        )

    raise LocalLLMError(
        f"Unknown LOCAL_LLM_PROVIDER={provider!r}. Use 'mlx' or 'transformers'."
    )


def generate_text_stream(system_prompt, user_prompt, token_budget=None, task=None, model=None):
    provider = provider_name()
    selected_model = model or model_name(task)
    if provider == "mlx":
        yield from _generate_mlx_stream(system_prompt, user_prompt, token_budget, selected_model)
        return

    yield generate_text(system_prompt, user_prompt, token_budget=token_budget, task=task, model=model)


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


def _generate_transformers(system_prompt, user_prompt, token_budget, selected_model):
    model, tokenizer, torch = _load_transformers_model(selected_model)
    prompt = _chat_prompt(tokenizer, system_prompt, user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=token_budget or max_tokens(),
            temperature=temperature(),
            do_sample=temperature() > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

import os
import re
import logging
import time
from functools import lru_cache

from backend.config import get_config

logger = logging.getLogger(__name__)


def _patch_transformers_auto_factory():
    """Fix transformers 5.x + Python 3.14 incompatibility.

    ``transformers.models.auto.auto_factory._LazyAutoMapping.register``
    calls ``key.__module__`` assuming *key* is always a class, but
    ``mlx_lm`` passes a plain string config name which causes
    ``AttributeError: 'str' object has no attribute '__module__'``.
    We wrap the method so it handles string keys gracefully.
    """
    try:
        from transformers.models.auto import auto_factory

        original_register = auto_factory._LazyAutoMapping.register

        def _safe_register(self, key, value, *, exist_ok=False):
            if isinstance(key, str):
                # Skip the __module__ check that crashes on string keys
                self._extra_content[key] = value
                return
            return original_register(self, key, value, exist_ok=exist_ok)

        auto_factory._LazyAutoMapping.register = _safe_register
        logger.info("Patched transformers auto_factory for Python 3.14 compat")
    except Exception:
        pass  # If transformers is not installed or API changed, skip silently


_patch_transformers_auto_factory()


class LocalLLMError(RuntimeError):
    pass





def provider_name():
    return get_config().llm_provider


def model_name(task=None):
    return get_config().model_for_task(task)


def task_model_config():
    config = get_config()
    return {
        "default": config.llm_model,
        "summary": config.llm_summary_model,
        "summary_deep": config.llm_summary_deep_model,
        "guidance": config.llm_guidance_model,
        "guidance_deep": config.llm_guidance_deep_model,
        "research_plan": config.llm_research_plan_model,
    }


def max_tokens(default=None):
    return default or get_config().llm_max_tokens


def temperature():
    return get_config().llm_temperature


@lru_cache(maxsize=4)
def _load_mlx_model(selected_model):
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise LocalLLMError(
            "MLX runtime is not installed. Install it with: pip install mlx-lm"
        ) from exc

    return load(selected_model)


@lru_cache(maxsize=2)
def _load_transformers_model(selected_model):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise LocalLLMError(
            "Transformers backend is not installed. Install torch and transformers "
            "before using this provider."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(selected_model)

    kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
    }

    if get_config().llm_load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Configured BitsAndBytes 4-bit quantization for %r", selected_model)
        except Exception as e:
            logger.warning(
                "Could not configure BitsAndBytes 4-bit quantization: %s. Loading in default precision.",
                e,
            )

    model = AutoModelForCausalLM.from_pretrained(
        selected_model,
        **kwargs,
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


class LLMService:
    """Provider-aware LLM facade.

    MLX is the supported default. ROCm/CUDA map to the Transformers backend so the
    rest of the app can switch providers by environment variable when it runs in
    an AMD/NVIDIA-capable environment.
    """

    def __init__(self, provider=None):
        self.provider = provider or provider_name()

    def generate(self, system_prompt, user_prompt, token_budget=None, task=None, model=None):
        selected_model = model or model_name(task)
        if self.provider == "fireworks":
            return _clean_response(
                _generate_fireworks(system_prompt, user_prompt, token_budget, selected_model)
            )
        if self.provider == "mlx":
            return _clean_response(
                _generate_mlx(system_prompt, user_prompt, token_budget, selected_model)
            )
        if self.provider == "transformers":
            return _clean_response(
                _generate_transformers(system_prompt, user_prompt, token_budget, selected_model)
            )
        raise LocalLLMError(
            f"Unknown LOCAL_LLM_PROVIDER={self.provider!r}. Use 'mlx', 'transformers', or 'fireworks'."
        )

    def stream(self, system_prompt, user_prompt, token_budget=None, task=None, model=None):
        selected_model = model or model_name(task)
        if self.provider == "fireworks":
            yield from _stream_fireworks(system_prompt, user_prompt, token_budget, selected_model)
            return
        if self.provider == "mlx":
            yield from _generate_mlx_stream(system_prompt, user_prompt, token_budget, selected_model)
            return
        if self.provider == "transformers":
            yield self.generate(system_prompt, user_prompt, token_budget, task, selected_model)
            return
        raise LocalLLMError(
            f"Unknown LOCAL_LLM_PROVIDER={self.provider!r}. Use 'mlx', 'transformers', or 'fireworks'."
        )


def generate_text(system_prompt, user_prompt, token_budget=None, task=None, model=None):
    return LLMService().generate(system_prompt, user_prompt, token_budget, task, model)


def generate_text_stream(system_prompt, user_prompt, token_budget=None, task=None, model=None):
    yield from LLMService().stream(system_prompt, user_prompt, token_budget, task, model)


def _clean_response(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.split(r"<end_of_turn>|<eos>|</s>", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = text.strip()

    # Strip raw headers like "Summary\n" or "Guidance\n" at the start
    text = re.sub(r"^(summary|guidance|research\s+plan):?\s*\n+", "", text, flags=re.IGNORECASE)

    # Process line-by-line to filter out echoes, planning, and constraint discussion
    cleaned_lines = []
    for line in text.splitlines():
        trimmed = line.strip()
        # Strip common robotic planning/thinking prefixes from the beginning of any line
        trimmed = re.sub(
            r"^(?:need\s+to\s+summarize|i['\u2019]ll\s+extract|we\s+need\s+to|so\s+the\s+paper|the\s+summary\s+should|i\s+need\s+to|let['\u2019]s\s+extract|we\s+are\s+asked|the\s+user\s+(?:wants|is\s+asking|requested)).*?:\s*",
            "",
            trimmed,
            flags=re.IGNORECASE
        )

        # Skip echoed prompt metadata blocks
        if re.match(r"^(?:the\s+)?paper\s+title:|^authors:|^abstract:|^evidence\s+\[\d+\]", trimmed, re.IGNORECASE):
            continue

        # Skip lines containing typical meta-commentary, reasoning, or prompt echo phrases
        if re.search(
            r"(?:the\s+summary\s+should|we\s+need\s+to|paper\s+says\s+that\s+i|so\s+the\s+paper|"
            r"no\s+bullet\s+points|just\s+what\s+the\s+paper\s+says|write\s+a\s+concise\s+summary|"
            r"avoid\s+hype|avoid\s+abstract-like|do\s+not\s+invent|do\s+not\s+use|do\s+not\s+give|"
            r"builder\s+recommendations|2-4\s+sentences|be\s+specific)",
            trimmed,
            re.IGNORECASE
        ):
            continue

        # Skip reasoning comments or constraint discussion
        if re.match(r"^(?:we\s+are\s+asked|the\s+user\s+(?:wants|is\s+asking|requested)|based\s+on\s+the\s+title\s+and\s+abstract|so\s+the\s+paper\s+proposes|the\s+summary\s+should\s+capture\s+that)", trimmed, re.IGNORECASE):
            continue
        # Skip short constraint notes / meta disclaimers
        if re.match(r"^\(that['\u2019]s|i\s+need\s+\d|i['\u2019]ll\s+mention|avoid\s+abstract-like|i['\u2019]ll\s+be", trimmed, re.IGNORECASE):
            continue

        # Strip transitional indicators like "I'll craft a summary: " or "Here is the summary: "
        trimmed = re.sub(
            r"^(?:i['\u2019]ll\s+craft\s+a\s+(?:concise\s+)?summary|i\s+will\s+craft\s+a\s+(?:concise\s+)?summary|here\s+is\s+a\s+(?:concise\s+)?summary|here(?:['\u2019]s|\s+is)\s+the\s+summary|summary):?\s*",
            "",
            trimmed,
            flags=re.IGNORECASE,
        )
        trimmed = re.sub(
            r"^(okay|sure|here(?:['\u2019]s|\s+is)),?\s+(?:a|an|the)?\s*",
            "",
            trimmed,
            flags=re.IGNORECASE,
        )
        if trimmed:
            cleaned_lines.append(trimmed)

    text = "\n".join(cleaned_lines).strip()

    heading_match = re.search(r"(\*\*[^*\n]+:\*\*|##\s+)", text)
    if heading_match and heading_match.start() < 280:
        preface = text[:heading_match.start()].lower()
        if any(marker in preface for marker in ("summary", "research plan", "formatted", "requested", "provided")):
            text = text[heading_match.start():].lstrip()
    fence_match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    return text.strip()


def _generate_fireworks(system_prompt, user_prompt, token_budget, selected_model):
    import requests
    import json
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('FIREWORKS_API_KEY', '')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": token_budget or max_tokens(),
        "temperature": temperature(),
    }
    start_time = time.perf_counter()
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    infer_time = time.perf_counter() - start_time
    logger.info("Fireworks model %r generated in %.2fs.", selected_model, infer_time)
    return response.json()["choices"][0]["message"]["content"].strip()


def _stream_fireworks(system_prompt, user_prompt, token_budget, selected_model):
    import requests
    import json
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('FIREWORKS_API_KEY', '')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": token_budget or max_tokens(),
        "temperature": temperature(),
        "stream": True
    }
    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data: ") and decoded_line != "data: [DONE]":
                try:
                    data = json.loads(decoded_line[6:])
                    content = data["choices"][0]["delta"].get("content")
                    if content:
                        yield content
                except Exception:
                    pass


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
    start_load = time.perf_counter()
    model, tokenizer, torch = _load_transformers_model(selected_model)
    load_time = time.perf_counter() - start_load

    prompt = _chat_prompt(tokenizer, system_prompt, user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    start_infer = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=token_budget or max_tokens(),
            temperature=temperature(),
            do_sample=temperature() > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    infer_time = time.perf_counter() - start_infer
    
    logger.info("Model %r loaded in %.2fs. Inference completed in %.2fs.", selected_model, load_time, infer_time)
    
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()




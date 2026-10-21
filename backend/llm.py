import os
import re
import logging
import time
from functools import lru_cache
from contextvars import ContextVar

from backend.config import get_config, env_number
from backend.errors import GenerationUnavailable, StreamInterrupted

logger = logging.getLogger(__name__)
generation_trace = ContextVar("generation_trace", default=None)


def trace_generation(provider, model, fallback=False):
    trace = generation_trace.get()
    if trace is not None:
        trace.update({"provider": provider, "model": model, "fallback": fallback})



class LocalLLMError(GenerationUnavailable):
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

    # Removed BitsAndBytes 4-bit quantization config because it causes issues on AMD ROCm setups.
    # We will load the model in standard precision (BF16 or FP16), which is faster and higher quality
    # given the VRAM available on the judging instances.

    model = AutoModelForCausalLM.from_pretrained(
        selected_model,
        **kwargs,
    )
    return model, tokenizer, torch


@lru_cache(maxsize=2)
def _load_mlx_model(selected_model):
    try:
        from backend.mlx_compat import import_mlx_lm
        load = import_mlx_lm().load
    except ImportError as exc:
        raise LocalLLMError(
            "MLX backend is not installed. Install mlx-lm before using this provider."
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


class LLMService:
    """Provider-aware LLM facade.

    MLX is the supported default. ROCm/CUDA map to the Transformers backend so the
    rest of the app can switch providers by environment variable when it runs in
    an AMD/NVIDIA-capable environment.
    """

    def __init__(self, provider=None):
        self.provider = provider or provider_name()

    def _route(self, selected_model, config):
        if selected_model.startswith("accounts/fireworks/") or self.provider == "fireworks":
            enabled = os.getenv("LOCAL_LLM_USE_FIREWORKS", "false").lower() == "true" or self.provider == "fireworks"
            if not enabled or not os.getenv("FIREWORKS_API_KEY"):
                raise LocalLLMError("Fireworks requires explicit opt-in and a key")
            return "fireworks"
        if config.remote_llm_enabled and selected_model == config.remote_llm_model:
            return "remote"
        return self.provider

    def _local(self, config, system_prompt, user_prompt, token_budget):
        # A cloud-only deployment has no authorized local model to recover to.
        if config.llm_provider not in {"mlx", "transformers"}:
            raise LocalLLMError("No local fallback configured")
        trace_generation(config.llm_provider, config.llm_model, fallback=True)
        function = _generate_mlx if config.llm_provider == "mlx" else _generate_transformers
        return function(system_prompt, user_prompt, token_budget, config.llm_model)

    def generate(self, system_prompt, user_prompt, token_budget=None, task=None, model=None):
        config = get_config()
        selected_model = model or model_name(task)
        route = self._route(selected_model, config)
        trace_generation(route, selected_model)
        try:
            if route == "remote":
                text = _generate_remote(system_prompt, user_prompt, token_budget, selected_model,
                                        config.remote_llm_base_url, config.remote_llm_api_key, config.remote_llm_provider)
            elif route == "fireworks":
                text = _generate_fireworks(system_prompt, user_prompt, token_budget, selected_model)
            elif route == "mlx":
                text = _generate_mlx(system_prompt, user_prompt, token_budget, selected_model)
            elif route == "transformers":
                text = _generate_transformers(system_prompt, user_prompt, token_budget, selected_model)
            else:
                raise LocalLLMError("Unknown generation provider")
            cleaned = _clean_response(text or "")
            if not cleaned:
                raise LocalLLMError("Provider returned an empty answer")
            return cleaned
        except Exception as exc:
            if route not in {"remote", "fireworks"}:
                raise LocalLLMError("Local generation failed") from exc
            logger.warning("Generation fallback provider=%s fallback=%s reason=%s", route, config.llm_provider, type(exc).__name__)
            try:
                cleaned = _clean_response(self._local(config, system_prompt, user_prompt, token_budget))
                if not cleaned:
                    raise LocalLLMError("Local fallback returned an empty answer")
                return cleaned
            except Exception as fallback_exc:
                raise LocalLLMError("Cloud generation and local recovery failed") from fallback_exc

    def stream(self, system_prompt, user_prompt, token_budget=None, task=None, model=None):
        config = get_config()
        selected_model = model or model_name(task)
        route = self._route(selected_model, config)
        trace_generation(route, selected_model)
        emitted = False
        try:
            if route == "remote":
                chunks = _generate_remote_stream(system_prompt, user_prompt, token_budget, selected_model,
                                                config.remote_llm_base_url, config.remote_llm_api_key, config.remote_llm_provider)
            elif route == "fireworks":
                chunks = _generate_fireworks_stream(system_prompt, user_prompt, token_budget, selected_model)
            elif route == "mlx":
                chunks = _generate_mlx_stream(system_prompt, user_prompt, token_budget, selected_model)
            else:
                chunks = iter([self.generate(system_prompt, user_prompt, token_budget, task, selected_model)])
            for chunk in chunks:
                if chunk:
                    emitted = True
                    yield chunk
            if not emitted:
                raise LocalLLMError("Provider returned an empty stream")
        except Exception as exc:
            if emitted:
                raise StreamInterrupted() from exc
            if route in {"remote", "fireworks"}:
                logger.warning("Stream fallback provider=%s fallback=%s reason=%s", route, config.llm_provider, type(exc).__name__)
                try:
                    result = _clean_response(self._local(config, system_prompt, user_prompt, token_budget))
                    if not result:
                        raise LocalLLMError("Local fallback returned an empty answer")
                    yield result
                    return
                except Exception as fallback_exc:
                    raise LocalLLMError("Cloud generation and local recovery failed") from fallback_exc
            raise LocalLLMError("Local stream failed") from exc


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


def _mlx_sampler():
    try:
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise LocalLLMError("Installed mlx-lm does not include sample_utils.make_sampler.") from exc

    return make_sampler(temp=temperature(), top_p=1.0, top_k=50)


def _generate_mlx(system_prompt, user_prompt, token_budget, selected_model):
    try:
        from backend.mlx_compat import import_mlx_lm
        generate = import_mlx_lm().generate
    except ImportError as exc:
        raise LocalLLMError(
            "MLX backend is not installed. Install mlx-lm before using this provider."
        ) from exc

    start_load = time.perf_counter()
    model, tokenizer = _load_mlx_model(selected_model)
    load_time = time.perf_counter() - start_load

    prompt = _chat_prompt(tokenizer, system_prompt, user_prompt)

    start_infer = time.perf_counter()
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=token_budget or max_tokens(),
        sampler=_mlx_sampler(),
        verbose=False,
    )
    infer_time = time.perf_counter() - start_infer

    logger.info("MLX model %r loaded in %.2fs. Inference completed in %.2fs.", selected_model, load_time, infer_time)
    return text.strip()


def _generate_mlx_stream(system_prompt, user_prompt, token_budget, selected_model):
    try:
        from backend.mlx_compat import import_mlx_lm
        stream_generate = import_mlx_lm().stream_generate
    except ImportError as exc:
        raise LocalLLMError(
            "MLX backend is not installed. Install mlx-lm before using this provider."
        ) from exc

    model, tokenizer = _load_mlx_model(selected_model)
    prompt = _chat_prompt(tokenizer, system_prompt, user_prompt)

    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=token_budget or max_tokens(),
        sampler=_mlx_sampler(),
    ):
        text = getattr(response, "text", response)
        if text:
            yield text


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

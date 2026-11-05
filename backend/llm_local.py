"""Local generation with Transformers/PyTorch or MLX."""
import logging
import time
from functools import lru_cache

from backend.llm_common import LocalLLMError, max_tokens, temperature

logger = logging.getLogger(__name__)


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

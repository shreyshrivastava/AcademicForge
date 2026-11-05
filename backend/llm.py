import os
import logging
from contextvars import ContextVar

from backend.config import get_config
from backend.errors import StreamInterrupted
from backend.llm_common import LocalLLMError
from backend.llm_local import _generate_mlx, _generate_mlx_stream, _generate_transformers
from backend.llm_remote import _generate_fireworks, _generate_fireworks_stream, _generate_remote, _generate_remote_stream
from backend.llm_text import _clean_response

logger = logging.getLogger(__name__)
generation_trace = ContextVar("generation_trace", default=None)


def trace_generation(provider, model, fallback=False):
    trace = generation_trace.get()
    if trace is not None:
        trace.update({"provider": provider, "model": model, "fallback": fallback})


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



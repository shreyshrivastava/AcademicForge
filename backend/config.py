import os
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


DEFAULT_PROVIDER = "mlx"
DEFAULT_MLX_MODEL = "mlx-community/gemma-2-2b-it-4bit"
DEFAULT_TRANSFORMERS_MODEL = "google/gemma-2-2b-it"
DEFAULT_MAX_TOKENS = 2500
DEFAULT_TEMPERATURE = 0.2

PROVIDER_ALIASES = {
    "mlx": "mlx",
    "mlxlm": "mlx",
    "mlx-lm": "mlx",
    "torch": "transformers",
    "transformer": "transformers",
    "transformers": "transformers",
}
try:
    LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
except ValueError:
    LLM_MAX_TOKENS = DEFAULT_MAX_TOKENS

try:
    LLM_TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
except ValueError:
    LLM_TEMPERATURE = DEFAULT_TEMPERATURE

@dataclass(frozen=True)
class AppConfig:
    llm_provider: str
    llm_model: str
    llm_summary_model: str
    llm_summary_deep_model: str
    llm_guidance_model: str
    llm_guidance_deep_model: str
    llm_research_plan_model: str
    llm_max_tokens: int
    llm_temperature: float
    llm_load_in_4bit: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        raw_provider = os.getenv("LOCAL_LLM_PROVIDER") or os.getenv("LLM_BACKEND") or DEFAULT_PROVIDER
        provider = raw_provider.strip().lower()
        provider = PROVIDER_ALIASES.get(provider, provider)

        if provider not in ("mlx", "transformers", "fireworks"):
            logger.warning("Unsupported local LLM provider %r. Defaulting to 'mlx'.", provider)
            provider = "mlx"

        if provider == "mlx":
            default_model = DEFAULT_MLX_MODEL
        elif provider == "fireworks":
            default_model = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/deepseek-v4-pro")
        else:
            default_model = DEFAULT_TRANSFORMERS_MODEL

        fast_model = os.getenv("LOCAL_LLM_MODEL", default_model)
        
        # Disable Fireworks Deep Mode if the API key is not present, fallback to local fast model
        default_deep = default_model
        if provider == "fireworks" and not os.getenv("FIREWORKS_API_KEY"):
            logger.warning("FIREWORKS_API_KEY is missing. Falling back to local model for Deep Mode.")
        else:
            default_deep = os.getenv("LOCAL_LLM_DEEP_MODEL", fast_model)
            
        deep_model = default_deep

        load_in_4bit = os.getenv("LOCAL_LLM_LOAD_IN_4BIT", "false").lower() == "true"

        return cls(
            llm_provider=provider,
            llm_model=fast_model,
            llm_summary_model=os.getenv("LOCAL_LLM_SUMMARY_MODEL", fast_model),
            llm_summary_deep_model=os.getenv("LOCAL_LLM_SUMMARY_DEEP_MODEL", deep_model),
            llm_guidance_model=os.getenv("LOCAL_LLM_GUIDANCE_MODEL", fast_model),
            llm_guidance_deep_model=os.getenv("LOCAL_LLM_GUIDANCE_DEEP_MODEL", deep_model),
            llm_research_plan_model=os.getenv("LOCAL_LLM_RESEARCH_PLAN_MODEL", deep_model),
            llm_max_tokens=LLM_MAX_TOKENS,
            llm_temperature=LLM_TEMPERATURE,
            llm_load_in_4bit=load_in_4bit,
        )

    def model_for_task_and_mode(self, task: str | None = None, mode: str | None = None) -> str:
        is_deep = (mode or "").strip().lower() == "deep"
        if task == "summary":
            return self.llm_summary_deep_model if is_deep else self.llm_summary_model
        if task == "guidance":
            return self.llm_guidance_deep_model if is_deep else self.llm_guidance_model
        if task == "research_plan":
            return self.llm_research_plan_model if is_deep else self.llm_model
        return self.llm_research_plan_model if is_deep else self.llm_model

    def model_for_task(self, task: str | None = None) -> str:
        return self.model_for_task_and_mode(task, "fast")

    def as_public_dict(self) -> dict:
        return {
            "llm_provider": self.llm_provider,
            "llm_models": {
                "default": self.llm_model,
                "summary": self.llm_summary_model,
                "summary_deep": self.llm_summary_deep_model,
                "guidance": self.llm_guidance_model,
                "guidance_deep": self.llm_guidance_deep_model,
                "research_plan": self.llm_research_plan_model,
            },
            "llm_max_tokens": self.llm_max_tokens,
            "llm_temperature": self.llm_temperature,
            "llm_load_in_4bit": self.llm_load_in_4bit,
        }


def get_config() -> AppConfig:
    return AppConfig.from_env()

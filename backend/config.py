import os
from dataclasses import dataclass
from pathlib import Path
import logging

from backend.runtime import auto_provider

logger = logging.getLogger(__name__)


DEFAULT_PROVIDER = "transformers"
DEFAULT_TRANSFORMERS_MODEL = "google/gemma-2-2b-it"
DEFAULT_MLX_MODEL = "mlx-community/gemma-2-2b-it-4bit"
DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
DEFAULT_MAX_TOKENS = 2500
DEFAULT_TEMPERATURE = 0.2
DEFAULT_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_JEV_MODEL = "jev-latest"
DEFAULT_SEARCH_MODE = "fast"
DEFAULT_REMOTE_PROVIDER = "groq"
DEFAULT_REMOTE_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_REMOTE_MODEL = "openai/gpt-oss-120b"

PROVIDER_ALIASES = {
    "amd": "transformers",
    "cuda": "transformers",
    "nvidia": "transformers",
    "apple": "mlx",
    "apple_silicon": "mlx",
    "auto": "auto",
    "metal": "mlx",
    "mlx-lm": "mlx",
    "mlxlm": "mlx",
    "mlx": "mlx",
    "rocm": "transformers",
    "pytorch": "transformers",
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
    remote_llm_enabled: bool
    remote_llm_provider: str
    remote_llm_base_url: str
    remote_llm_model: str
    remote_llm_api_key: str
    search_mode: str
    jev_enabled: bool
    jev_api_key: str
    jev_endpoint: str
    jev_model: str
    jev_timeout_seconds: float
    jev_include_details: bool
    jev_max_papers: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        raw_provider = auto_provider()
        provider = raw_provider.strip().lower()
        provider = PROVIDER_ALIASES.get(provider, provider)
        if provider == "auto":
            provider = DEFAULT_PROVIDER

        if provider not in ("transformers", "fireworks", "mlx"):
            logger.warning("Unsupported local LLM provider %r. Defaulting to 'transformers'.", provider)
            provider = "transformers"

        if provider == "fireworks":
            default_model = os.getenv("FIREWORKS_MODEL", DEFAULT_FIREWORKS_MODEL)
        elif provider == "mlx":
            default_model = DEFAULT_MLX_MODEL
        else:
            default_model = DEFAULT_TRANSFORMERS_MODEL

        fast_model = os.getenv("LOCAL_LLM_MODEL", default_model)
        
        fireworks_model = os.getenv("FIREWORKS_MODEL", DEFAULT_FIREWORKS_MODEL)
        use_fireworks = os.getenv("LOCAL_LLM_USE_FIREWORKS", "false").lower() == "true"
        remote_provider = os.getenv("REMOTE_LLM_PROVIDER", DEFAULT_REMOTE_PROVIDER).strip().lower()
        remote_base_url = os.getenv("REMOTE_LLM_BASE_URL", DEFAULT_REMOTE_BASE_URL)
        remote_model = os.getenv("REMOTE_LLM_MODEL", DEFAULT_REMOTE_MODEL)
        remote_api_key = os.getenv("REMOTE_LLM_API_KEY") or os.getenv("GROQ_API_KEY", "")
        remote_requested = os.getenv("REMOTE_LLM_USE_FOR_DEEP", "false").lower() == "true"
        remote_enabled = bool(remote_requested and remote_api_key)

        if remote_enabled:
            default_deep = remote_model
        elif os.getenv("FIREWORKS_API_KEY") and (provider == "fireworks" or use_fireworks):
            default_deep = fireworks_model
        else:
            if provider == "fireworks" or use_fireworks:
                logger.warning("FIREWORKS_API_KEY is missing. Falling back to local model for Deep Mode.")
            elif remote_requested:
                logger.warning("Remote Deep Mode key is missing. Falling back to local model.")
            default_deep = fast_model
            
        deep_model = os.getenv("LOCAL_LLM_DEEP_MODEL", default_deep)

        load_in_4bit = os.getenv("LOCAL_LLM_LOAD_IN_4BIT", "false").lower() == "true"
        search_mode = os.getenv("ACADEMICFORGE_SEARCH_MODE", DEFAULT_SEARCH_MODE).strip().lower()
        if search_mode not in {"fast", "quality"}:
            logger.warning("Unsupported search mode %r. Defaulting to 'fast'.", search_mode)
            search_mode = DEFAULT_SEARCH_MODE
        jev_enabled = (
            search_mode == "quality"
            and os.getenv("JEV_ENABLED", "false").lower() == "true"
        )
        jev_include_details = os.getenv("JEV_INCLUDE_DETAILS", "false").lower() == "true"
        try:
            jev_max_papers = max(1, int(os.getenv("JEV_MAX_PAPERS", "3")))
        except ValueError:
            jev_max_papers = 3
        try:
            jev_timeout_seconds = float(os.getenv("JEV_TIMEOUT_SECONDS", "60"))
        except ValueError:
            jev_timeout_seconds = 60.0

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
            remote_llm_enabled=remote_enabled,
            remote_llm_provider=remote_provider,
            remote_llm_base_url=remote_base_url,
            remote_llm_model=remote_model,
            remote_llm_api_key=remote_api_key,
            search_mode=search_mode,
            jev_enabled=jev_enabled,
            jev_api_key=os.getenv("TYPESAFE_API_KEY", ""),
            jev_endpoint=os.getenv("JEV_ENDPOINT", DEFAULT_JEV_ENDPOINT),
            jev_model=os.getenv("JEV_MODEL", DEFAULT_JEV_MODEL),
            jev_timeout_seconds=jev_timeout_seconds,
            jev_include_details=jev_include_details,
            jev_max_papers=jev_max_papers,
        )

    def model_for_task_and_mode(self, task: str | None = None, mode: str | None = None) -> str:
        is_deep = (mode or "").strip().lower() == "deep"
        if task == "summary":
            return self.llm_summary_deep_model if is_deep else self.llm_summary_model
        if task == "guidance":
            return self.llm_guidance_deep_model if is_deep else self.llm_guidance_model
        if task == "research_plan":
            return self.llm_research_plan_model
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
            "remote_llm_enabled": self.remote_llm_enabled,
            "remote_llm_provider": self.remote_llm_provider,
            "remote_llm_model": self.remote_llm_model,
            "search_mode": self.search_mode,
            "jev_enabled": self.jev_enabled and bool(
                self.jev_api_key or self.jev_endpoint.startswith(("http://127.0.0.1", "http://localhost"))
            ),
            "jev_model": self.jev_model,
        }


def get_config() -> AppConfig:
    return AppConfig.from_env()

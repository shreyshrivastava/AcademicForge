import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROVIDER = "mlx"
DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-4bit"
DEFAULT_MAX_TOKENS = 700
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_CACHE_MAX_FILES = 500

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

CACHE_DIR = Path(os.getenv("ACADEMICFORGE_CACHE_DIR", ".academicforge_cache"))

try:
    CACHE_TTL_SECONDS = int(os.getenv("ACADEMICFORGE_CACHE_TTL_SECONDS", str(DEFAULT_CACHE_TTL_SECONDS)))
except ValueError:
    CACHE_TTL_SECONDS = DEFAULT_CACHE_TTL_SECONDS

try:
    CACHE_MAX_FILES = int(os.getenv("ACADEMICFORGE_CACHE_MAX_FILES", str(DEFAULT_CACHE_MAX_FILES)))
except ValueError:
    CACHE_MAX_FILES = DEFAULT_CACHE_MAX_FILES


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
    cache_dir: Path
    cache_ttl_seconds: int
    cache_max_files: int
    llm_load_in_4bit: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        raw_provider = os.getenv("LOCAL_LLM_PROVIDER") or os.getenv("LLM_BACKEND") or DEFAULT_PROVIDER
        provider = raw_provider.strip().lower()
        provider = PROVIDER_ALIASES.get(provider, provider)

        if provider not in ("mlx", "transformers"):
            logger.warning("Unsupported local LLM provider %r. Defaulting to 'mlx'.", provider)
            provider = "mlx"

        default_model = "mlx-community/gemma-4-e2b-it-4bit" if provider == "mlx" else "google/gemma-4-e2b-it"
        default_deep_model = "mlx-community/gemma-4-e2b-it-OptiQ-4bit" if provider == "mlx" else "google/gemma-4-31b-it"
        load_in_4bit = os.getenv("LOCAL_LLM_LOAD_IN_4BIT", "false").lower() == "true"

        return cls(
            llm_provider=provider,
            llm_model=default_model,
            llm_summary_model=default_model,
            llm_summary_deep_model=default_deep_model,
            llm_guidance_model=default_model,
            llm_guidance_deep_model=default_deep_model,
            llm_research_plan_model=default_deep_model,
            llm_max_tokens=LLM_MAX_TOKENS,
            llm_temperature=LLM_TEMPERATURE,
            cache_dir=CACHE_DIR,
            cache_ttl_seconds=CACHE_TTL_SECONDS,
            cache_max_files=CACHE_MAX_FILES,
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
            "cache": {
                "dir": str(self.cache_dir),
                "ttl_seconds": self.cache_ttl_seconds,
                "max_files": self.cache_max_files,
            },
        }


def get_config() -> AppConfig:
    return AppConfig.from_env()

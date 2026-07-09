import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROVIDER = "mlx"
DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"
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
    "cuda": "transformers",
    "rocm": "transformers",
    "roc": "transformers",
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
    llm_research_plan_model: str
    llm_max_tokens: int
    llm_temperature: float
    cache_dir: Path
    cache_ttl_seconds: int
    cache_max_files: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        raw_provider = os.getenv("LOCAL_LLM_PROVIDER") or os.getenv("LLM_BACKEND") or DEFAULT_PROVIDER
        provider = raw_provider.strip().lower()
        provider = PROVIDER_ALIASES.get(provider, provider)

        raw_model = os.getenv("LOCAL_LLM_MODEL")
        if raw_model:
            default_model = raw_model.strip()
        else:
            if provider == "openai":
                default_model = "gpt-4o-mini"
            elif provider == "fireworks":
                default_model = "accounts/fireworks/models/qwen2p5-8b-instruct"
            elif provider == "transformers":
                default_model = "Qwen/Qwen2.5-7B-Instruct"
            elif provider in {"qwen", "dashscope"}:
                default_model = "qwen-plus"
            else:
                default_model = DEFAULT_MODEL

        return cls(
            llm_provider=provider,
            llm_model=default_model,
            llm_summary_model=os.getenv("LOCAL_LLM_SUMMARY_MODEL", default_model).strip(),
            llm_research_plan_model=(
                os.getenv("LOCAL_LLM_RESEARCH_PLAN_MODEL")
                or default_model
            ).strip(),
            llm_max_tokens=LLM_MAX_TOKENS,
            llm_temperature=LLM_TEMPERATURE,
            cache_dir=CACHE_DIR,
            cache_ttl_seconds=CACHE_TTL_SECONDS,
            cache_max_files=CACHE_MAX_FILES,
        )

    def model_for_task(self, task: str | None = None) -> str:
        if task == "summary":
            return self.llm_summary_model
        if task == "research_plan":
            return self.llm_research_plan_model
        return self.llm_model

    def as_public_dict(self) -> dict:
        return {
            "llm_provider": self.llm_provider,
            "llm_models": {
                "default": self.llm_model,
                "summary": self.llm_summary_model,
                "research_plan": self.llm_research_plan_model,
            },
            "llm_max_tokens": self.llm_max_tokens,
            "llm_temperature": self.llm_temperature,
            "cache": {
                "dir": str(self.cache_dir),
                "ttl_seconds": self.cache_ttl_seconds,
                "max_files": self.cache_max_files,
            },
        }


def get_config() -> AppConfig:
    return AppConfig.from_env()

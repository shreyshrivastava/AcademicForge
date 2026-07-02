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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _provider_from_env() -> str:
    raw_provider = (
        os.getenv("LOCAL_LLM_PROVIDER")
        or os.getenv("LLM_BACKEND")
        or DEFAULT_PROVIDER
    )
    provider = raw_provider.strip().lower()
    return PROVIDER_ALIASES.get(provider, provider)


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str
    llm_model: str
    llm_summary_model: str
    llm_roadmap_model: str
    llm_max_tokens: int
    llm_temperature: float
    cache_dir: Path
    cache_ttl_seconds: int
    cache_max_files: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        default_model = os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL).strip()
        return cls(
            llm_provider=_provider_from_env(),
            llm_model=default_model,
            llm_summary_model=os.getenv("LOCAL_LLM_SUMMARY_MODEL", default_model).strip(),
            llm_roadmap_model=os.getenv("LOCAL_LLM_ROADMAP_MODEL", default_model).strip(),
            llm_max_tokens=_env_int("LOCAL_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
            llm_temperature=_env_float("LOCAL_LLM_TEMPERATURE", DEFAULT_TEMPERATURE),
            cache_dir=Path(os.getenv("ACADEMICFORGE_CACHE_DIR", ".academicforge_cache")),
            cache_ttl_seconds=_env_int("ACADEMICFORGE_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS),
            cache_max_files=_env_int("ACADEMICFORGE_CACHE_MAX_FILES", DEFAULT_CACHE_MAX_FILES),
        )

    def model_for_task(self, task: str | None = None) -> str:
        if task == "summary":
            return self.llm_summary_model
        if task == "roadmap":
            return self.llm_roadmap_model
        return self.llm_model

    def as_public_dict(self) -> dict:
        return {
            "llm_provider": self.llm_provider,
            "llm_models": {
                "default": self.llm_model,
                "summary": self.llm_summary_model,
                "roadmap": self.llm_roadmap_model,
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

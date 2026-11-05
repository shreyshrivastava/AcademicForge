"""Settings and errors shared by the local and remote generation backends."""

from backend.config import get_config
from backend.errors import GenerationUnavailable


class LocalLLMError(GenerationUnavailable):
    pass


def max_tokens(default=None):
    return default or get_config().llm_max_tokens


def temperature():
    return get_config().llm_temperature

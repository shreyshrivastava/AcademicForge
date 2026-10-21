"""Isolate mlx-lm's legacy tokenizer registration from Transformers 5.

mlx-lm registers its NewlineTokenizer with a string config, which Transformers
5 rejects. The Gemma path does not use this tokenizer. Skip only that obsolete
registration during import, then immediately restore the public method.
"""

import importlib
import logging
from threading import Lock

_lock = Lock()
logger = logging.getLogger(__name__)


def import_mlx_lm():
    with _lock:
        from transformers import AutoTokenizer

        descriptor = AutoTokenizer.__dict__["register"]
        original = AutoTokenizer.register

        def register(config_class, *args, **kwargs):
            if config_class == "NewlineTokenizer":
                logger.warning("Skipping legacy mlx-lm NewlineTokenizer registration; using model tokenizer")
                return
            return original(config_class, *args, **kwargs)

        AutoTokenizer.register = staticmethod(register)
        try:
            return importlib.import_module("mlx_lm")
        finally:
            AutoTokenizer.register = descriptor

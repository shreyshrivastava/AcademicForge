"""Explicit component readiness; lexical retrieval is a supported degraded mode."""

import logging
from threading import Lock

from backend.config import get_config

logger = logging.getLogger(__name__)
_lock = Lock()
_components = {name: "pending" for name in ("generation", "dense", "reranker")}


def warmup():
    from backend.llm import _load_mlx_model, _load_transformers_model
    from backend.retrieval.dense import _load_sentence_transformer
    from backend.retrieval.reranker import get_reranker_model

    config = get_config()
    loaders = {
        "generation": lambda: (_load_mlx_model if config.llm_provider == "mlx" else _load_transformers_model)(config.llm_model)
        if config.llm_provider != "fireworks" else True,
        "dense": _load_sentence_transformer,
        "reranker": get_reranker_model,
    }
    for name, loader in loaders.items():
        try:
            loaded = loader()
            state = "ready" if loaded is not None else "unavailable"
        except Exception:
            logger.exception("Component warmup failed component=%s", name)
            state = "unavailable"
        with _lock:
            _components[name] = state


def readiness():
    with _lock:
        components = dict(_components)
    ready = components["generation"] == "ready"
    if "pending" in components.values():
        status = "warming_up"
    elif not ready:
        status = "unavailable"
    elif any(state != "ready" for state in components.values()):
        status = "degraded"
    else:
        status = "ok"
    return {"status": status, "models_ready": ready, "components": components}

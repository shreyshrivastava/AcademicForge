"""Small in-memory cache for completed paper searches."""

from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import json
import os
import time


_MAX_ENTRIES = 64
_CACHE: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()


def _ttl_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("ACADEMICFORGE_SEARCH_CACHE_TTL_SECONDS", "300")))
    except ValueError:
        return 300.0


def cache_key(query: str, categories: list[str] | None, search_mode: str) -> str:
    state = {
        "query": " ".join(query.split()).lower(),
        "categories": sorted(categories or []),
        "search_mode": search_mode,
    }
    encoded = json.dumps(state, sort_keys=True, ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def get_cached(key: str) -> list[dict] | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    created_at, papers = entry
    if time.monotonic() - created_at >= _ttl_seconds():
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return deepcopy(papers)


def store(key: str, papers: list[dict]) -> None:
    _CACHE[key] = (time.monotonic(), deepcopy(papers))
    _CACHE.move_to_end(key)
    while len(_CACHE) > _MAX_ENTRIES:
        _CACHE.popitem(last=False)


def clear() -> None:
    _CACHE.clear()

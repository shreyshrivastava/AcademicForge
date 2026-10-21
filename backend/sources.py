"""Bounded paper-source HTTP calls and per-search diagnostics."""

from contextvars import ContextVar
from copy import deepcopy
import logging
import time

import requests

from backend.errors import SourceUnavailable

logger = logging.getLogger(__name__)
search_diagnostics = ContextVar("search_diagnostics", default=None)
search_deadline = ContextVar("search_deadline", default=0.0)


def begin_search():
    search_diagnostics.set({"sources": {}, "warnings": []})
    search_deadline.set(time.monotonic() + 40)


def diagnostics():
    return deepcopy(search_diagnostics.get() or {"sources": {}, "warnings": []})


def source_status(name, status):
    state = diagnostics()
    state["sources"][name] = status
    if status == "unavailable":
        warning = f"{name} is unavailable; results may be incomplete."
        if warning not in state["warnings"]:
            state["warnings"].append(warning)
    search_diagnostics.set(state)


def get_source(url, **kwargs):
    deadline = search_deadline.get() or time.monotonic() + 20
    for attempt in range(2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SourceUnavailable()
        response = requests.get(url, timeout=(min(3, remaining), min(8, remaining)), **kwargs)
        if response.status_code not in {429, 502, 503, 504} or attempt:
            response.raise_for_status()
            return response
        try:
            delay = max(0, float(response.headers.get("Retry-After", "0.5")))
        except ValueError:
            delay = 0.5
        response.close()
        # Do not violate a long Retry-After or exhaust the whole search deadline.
        if delay > 1 or time.monotonic() + delay >= deadline:
            raise SourceUnavailable()
        time.sleep(delay)
    raise SourceUnavailable()

import hashlib
import json
import os
from pathlib import Path
from typing import Any


CACHE_DIR = Path(os.getenv("ACADEMICFORGE_CACHE_DIR", ".academicforge_cache"))


def make_cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_get(namespace: str, key: str):
    path = _cache_path(namespace, key)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))["value"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def cache_set(namespace: str, key: str, value):
    path = _cache_path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps({"value": value}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _cache_path(namespace: str, key: str) -> Path:
    safe_namespace = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in namespace
    )
    return CACHE_DIR / safe_namespace / f"{key}.json"

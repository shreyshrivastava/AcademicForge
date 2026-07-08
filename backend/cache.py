import hashlib
import json
import time
from pathlib import Path
from typing import Any

from backend.config import get_config

CACHE_DIR = get_config().cache_dir


def make_cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_get(namespace: str, key: str):
    path = _cache_path(namespace, key)
    if not path.exists():
        return None

    if _is_expired(path):
        try:
            path.unlink()
        except OSError:
            pass
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
    _prune_namespace(path.parent)


def _cache_path(namespace: str, key: str) -> Path:
    safe_namespace = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in namespace
    )
    return CACHE_DIR / safe_namespace / f"{key}.json"


def _is_expired(path: Path) -> bool:
    ttl_seconds = get_config().cache_ttl_seconds
    if ttl_seconds <= 0:
        return False
    try:
        return time.time() - path.stat().st_mtime > ttl_seconds
    except OSError:
        return True


def _prune_namespace(namespace_dir: Path) -> None:
    max_files = get_config().cache_max_files
    if max_files <= 0:
        return
    try:
        files = sorted(
            namespace_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for old_file in files[max_files:]:
        try:
            old_file.unlink()
        except OSError:
            pass


def check_and_increment_usage(limit: int = 10) -> int:
    """Read usage_counter.json from cache folder and increment it.

    If the count meets or exceeds the limit, raises a RuntimeError.
    """
    import os
    if os.getenv("DISABLE_DEMO_LIMIT", "").lower() == "true" or os.getenv("DISABLE_USAGE_LIMIT", "").lower() == "true":
        return 0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    counter_path = CACHE_DIR / "usage_counter.json"

    count = 0
    if counter_path.exists():
        try:
            data = json.loads(counter_path.read_text(encoding="utf-8"))
            count = data.get("usage_count", 0)
        except Exception:
            pass

    if count >= limit:
        raise RuntimeError(
            f"Demo credit limit reached ({limit} generations used). "
            "Host your own instance of AcademicForge with your API key to run more queries."
        )

    count += 1
    try:
        counter_path.write_text(json.dumps({"usage_count": count}), encoding="utf-8")
    except Exception:
        pass

    return count


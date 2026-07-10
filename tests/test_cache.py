import os
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.cache as cache


def test_answer_cache_disabled_by_default():
    original_cache_flag = os.environ.pop("ACADEMICFORGE_ENABLE_ANSWER_CACHE", None)
    try:
        key = cache.make_cache_key("namespace", {"b": 2, "a": 1})
        cache.cache_set("unit", key, {"answer": 42})
        assert cache.cache_get("unit", key) is None
    finally:
        if original_cache_flag is not None:
            os.environ["ACADEMICFORGE_ENABLE_ANSWER_CACHE"] = original_cache_flag


def test_cache_round_trip_when_enabled():
    original_cache_dir = cache.CACHE_DIR
    original_cache_flag = os.environ.get("ACADEMICFORGE_ENABLE_ANSWER_CACHE")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.environ["ACADEMICFORGE_ENABLE_ANSWER_CACHE"] = "true"
            cache.CACHE_DIR = Path(tmpdir)
            key = cache.make_cache_key("namespace", {"b": 2, "a": 1})
            assert cache.cache_get("unit", key) is None
            cache.cache_set("unit", key, {"answer": 42})
            assert cache.cache_get("unit", key) == {"answer": 42}
        finally:
            cache.CACHE_DIR = original_cache_dir
            if original_cache_flag is None:
                os.environ.pop("ACADEMICFORGE_ENABLE_ANSWER_CACHE", None)
            else:
                os.environ["ACADEMICFORGE_ENABLE_ANSWER_CACHE"] = original_cache_flag


def test_cache_key_is_stable_for_dict_order():
    left = cache.make_cache_key({"a": 1, "b": 2})
    right = cache.make_cache_key({"b": 2, "a": 1})
    assert left == right


if __name__ == "__main__":
    test_answer_cache_disabled_by_default()
    test_cache_round_trip_when_enabled()
    test_cache_key_is_stable_for_dict_order()
    print("cache tests passed")

import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backend.cache as cache


def test_cache_round_trip():
    original_cache_dir = cache.CACHE_DIR
    with tempfile.TemporaryDirectory() as tmpdir:
        cache.CACHE_DIR = Path(tmpdir)
        key = cache.make_cache_key("namespace", {"b": 2, "a": 1})
        assert cache.cache_get("unit", key) is None
        cache.cache_set("unit", key, {"answer": 42})
        assert cache.cache_get("unit", key) == {"answer": 42}
    cache.CACHE_DIR = original_cache_dir


def test_cache_key_is_stable_for_dict_order():
    left = cache.make_cache_key({"a": 1, "b": 2})
    right = cache.make_cache_key({"b": 2, "a": 1})
    assert left == right


if __name__ == "__main__":
    test_cache_round_trip()
    test_cache_key_is_stable_for_dict_order()
    print("cache tests passed")

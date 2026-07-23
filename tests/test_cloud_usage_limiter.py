from pathlib import Path

from frontend.cloud_usage_limiter import consume_one_request, hash_client_identifier


def test_hash_client_identifier_does_not_store_raw_ip():
    digest = hash_client_identifier("203.0.113.10", "test-salt")

    assert "203.0.113.10" not in digest
    assert len(digest) == 64


def test_consume_one_request_allows_first_and_blocks_second(tmp_path: Path):
    db_path = tmp_path / "usage.sqlite3"

    first = consume_one_request(
        "203.0.113.10",
        salt="test-salt",
        db_path=db_path,
        request_limit=1,
    )
    second = consume_one_request(
        "203.0.113.10",
        salt="test-salt",
        db_path=db_path,
        request_limit=1,
    )

    assert first.allowed is True
    assert second.allowed is False
    assert first.client_hash == second.client_hash

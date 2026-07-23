"""Best-effort one-request limiter for the public Streamlit Cloud demo."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB_PATH = Path(".academicforge_cloud/usage.sqlite3")


@dataclass(frozen=True)
class UsageDecision:
    allowed: bool
    client_hash: str
    reason: str


def hash_client_identifier(client_identifier: str, salt: str) -> str:
    value = f"{salt}:{client_identifier}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def consume_one_request(
    client_identifier: str,
    *,
    salt: str,
    db_path: Path = DEFAULT_DB_PATH,
    request_limit: int = 1,
) -> UsageDecision:
    """Record one public-demo request without storing the raw IP address."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    client_hash = hash_client_identifier(client_identifier, salt)
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cloud_usage (
                client_hash TEXT PRIMARY KEY,
                request_count INTEGER NOT NULL,
                first_seen_utc TEXT NOT NULL,
                last_seen_utc TEXT NOT NULL
            )
            """
        )
        row = conn.execute(
            "SELECT request_count FROM cloud_usage WHERE client_hash = ?",
            (client_hash,),
        ).fetchone()
        if row and int(row[0]) >= request_limit:
            conn.execute(
                "UPDATE cloud_usage SET last_seen_utc = ? WHERE client_hash = ?",
                (now, client_hash),
            )
            return UsageDecision(
                allowed=False,
                client_hash=client_hash,
                reason="This IP address has already used the public cloud request.",
            )

        if row:
            conn.execute(
                """
                UPDATE cloud_usage
                SET request_count = request_count + 1, last_seen_utc = ?
                WHERE client_hash = ?
                """,
                (now, client_hash),
            )
        else:
            conn.execute(
                """
                INSERT INTO cloud_usage (
                    client_hash, request_count, first_seen_utc, last_seen_utc
                )
                VALUES (?, 1, ?, ?)
                """,
                (client_hash, now, now),
            )

    return UsageDecision(
        allowed=True,
        client_hash=client_hash,
        reason="Request accepted.",
    )


def reset_usage_store(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Test helper for clearing the local usage database."""
    if db_path.exists():
        db_path.unlink()


def limiter_salt(default: str = "academicforge-cloud-demo") -> str:
    return os.getenv("ACADEMICFORGE_LIMITER_SALT", default)

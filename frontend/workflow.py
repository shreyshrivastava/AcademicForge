"""Pure cache identities for generated paper outputs."""

from hashlib import sha256
import json

PROMPT_VERSION = "2026-10-21-v1"


def paper_output_key(paper, mode, models):
    # Source content, task models, mode and prompt revision all affect the answer.
    state = {"paper": {key: paper.get(key) for key in (
        "paper_id", "title", "abstract", "authors", "date", "url", "link", "metadata",
    )}, "mode": mode, "models": models, "prompt_version": PROMPT_VERSION}
    return sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()

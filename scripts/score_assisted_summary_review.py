"""Summarize an AI-assisted human-adjudication summary review."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = ROOT / "docs" / "benchmarks" / "assisted_summary_review.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "benchmarks" / "assisted_summary_review_pilot.json"
RATING_FIELDS = (
    "output_a_faithfulness_1_to_5",
    "output_a_usefulness_1_to_5",
    "output_b_faithfulness_1_to_5",
    "output_b_usefulness_1_to_5",
)


def summarize_rows(rows: list[dict]) -> dict:
    item_ids = [row["item_id"].strip() for row in rows]
    if not all(item_ids) or len(item_ids) != len(set(item_ids)):
        raise ValueError("Every review item needs a unique item_id")
    better = [row["better_output"].strip().upper() for row in rows if row["better_output"].strip()]
    if set(better) - {"A", "B", "TIE"}:
        raise ValueError("better_output must be A, B, or TIE")
    return {
        "items": len(rows),
        "better_output_counts": dict(sorted(Counter(better).items())),
        "numeric_rating_items": sum(all(row[field].strip() for field in RATING_FIELDS) for row in rows),
        "adjudication_counts": dict(sorted(Counter(row["user_adjudication"].strip() for row in rows if row["user_adjudication"].strip()).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.review.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    summary = summarize_rows(rows)
    artifact = {
        "evaluation": "AcademicForge generated-summary AI-assisted human-adjudication pilot",
        "classification": "not a blinded or independent human evaluation",
        "review_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "summary": summary,
        "limits": [
            "The AI supplied evidence analysis before the user selected outputs.",
            "The user reported parallel web research, but source URLs were not logged.",
            "Numeric ratings are incomplete and must not be compared to the blinded human-evaluation score.",
            "The ten items are an exploratory sample, not a general performance estimate.",
        ],
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

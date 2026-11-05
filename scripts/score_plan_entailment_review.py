"""Validate and summarize a local Research Plan claim-entailment review."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from benchmarks.common import ROOT, file_sha256


DEFAULT_RATINGS = ROOT / "docs" / "benchmarks" / "plan_entailment_ratings.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "benchmarks" / "plan_entailment_pilot.json"
LABELS = {"entailed", "partially_supported", "insufficient", "contradicted", "unclear"}


def _lane(rows: list[dict], key: str) -> dict:
    labels = [row[key].strip() for row in rows if row[key].strip()]
    invalid = sorted(set(labels) - LABELS)
    if invalid:
        raise ValueError(f"Unsupported {key} label(s): {', '.join(invalid)}")
    counts = dict(sorted(Counter(labels).items()))
    return {"rated_claims": len(labels), "counts": counts}


def summarize_rows(rows: list[dict]) -> dict:
    finding_ids = [row["finding_id"].strip() for row in rows]
    if any(not finding_id for finding_id in finding_ids):
        raise ValueError("Every row requires a finding_id")
    if len(finding_ids) != len(set(finding_ids)):
        raise ValueError("The review contains duplicate finding IDs")
    return {
        "claims_in_sheet": len(rows),
        "citation_entailment": _lane(rows, "citation_entailment_label"),
        "external_fact_check": _lane(rows, "external_fact_check_label"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.ratings.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    summary = summarize_rows(rows)
    artifact = {
        "evaluation": "AcademicForge Research Plan claim-entailment pilot",
        "classification": "exploratory single-reviewer pilot; not an expert or independent factuality evaluation",
        "ratings_sha256": file_sha256(args.ratings),
        "reviewer_context": "project-owner, non-domain-expert; citation reading was assisted with source summaries",
        "rubric": {
            "entailed": "The cited evidence directly supports the claim as written.",
            "partially_supported": "The evidence supports the core idea, but the claim is overgeneralized or omits a material limitation.",
            "insufficient": "The cited evidence does not provide enough support for the claim as written.",
            "contradicted": "The cited evidence supports the opposite of the claim.",
            "unclear": "The evidence is not clear enough for the reviewer to decide.",
        },
        "summary": summary,
        "limits": [
            "The 10 claims are an exploratory subset, not a prevalence estimate for all generated plans.",
            "Citation entailment and external fact checking are separate lanes and must not be combined.",
            "External checks cover only claims with recorded external sources.",
            "No inter-rater agreement or domain-expert validation was performed.",
        ],
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

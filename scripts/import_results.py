"""Validate a GPU ablation artifact and refresh the published ablation tables."""

import argparse
import json
from pathlib import Path
import re

from benchmarks.run_scifact import DATASETS, SYSTEMS, source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
README_PATHS = (ROOT / "README.md", ROOT / "docs" / "benchmarks" / "README.md")
START = "<!-- BEGIN GENERATED ABLATION TABLE -->"
END = "<!-- END GENERATED ABLATION TABLE -->"


def load_and_validate(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark") != "AcademicForge retrieval ablation":
        raise ValueError("Not an AcademicForge retrieval ablation artifact")
    if payload.get("source_sha256") != source_fingerprint():
        raise ValueError("Artifact source fingerprint does not match this checkout")
    reports = payload.get("datasets")
    if not isinstance(reports, list) or not reports:
        raise ValueError("Artifact has no dataset reports")
    for report in reports:
        dataset_id = report.get("dataset_id")
        expected = DATASETS.get(dataset_id)
        if expected is None or report.get("dataset_sha256") != expected["sha256"]:
            raise ValueError(f"Dataset checksum validation failed for {dataset_id!r}")
        if not all(system in report.get("systems", {}) for system in SYSTEMS):
            raise ValueError(f"Artifact is missing one or more systems for {dataset_id!r}")
    return payload


def format_metric(system: dict, metric: str) -> str:
    details = system["metrics"][metric]
    return f"{details['mean']:.4f} ({details['ci95_low']:.4f}-{details['ci95_high']:.4f})"


def table_for(payload: dict) -> str:
    classification = payload["result_classification"]
    heading = "Imported benchmark result" if payload.get("is_full_benchmark") else f"Imported {classification} result"
    lines = [
        START,
        f"**{heading}.** Values are nDCG@10 (95% query-bootstrap CI).",
        "",
        "| Dataset | BM25 | Stemmed BM25 | BGE dense | RRF | RRF + cross-encoder |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in payload["datasets"]:
        values = [format_metric(report["systems"][system], "ndcg@10") for system in SYSTEMS]
        lines.append(f"| {report['dataset']} | " + " | ".join(values) + " |")
    if not payload.get("is_full_benchmark"):
        labels = ", ".join(payload.get("result_labels", []))
        lines.extend(["", f"This is labeled **{labels}** and must not be presented as the full benchmark result."])
    lines.append(END)
    return "\n".join(lines)


def replace_table(path: Path, table: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.escape(START) + r".*?" + re.escape(END)
    updated, count = re.subn(pattern, table, text, flags=re.DOTALL)
    if count != 1:
        raise ValueError(f"Expected exactly one generated ablation table marker pair in {path}")
    path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="GPU-produced retrieval-ablation.json")
    args = parser.parse_args()
    payload = load_and_validate(args.results)
    table = table_for(payload)
    for path in README_PATHS:
        replace_table(path, table)
    print(f"Validated {args.results} and updated {len(README_PATHS)} README ablation tables.")


if __name__ == "__main__":
    main()

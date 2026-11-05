"""Build a local, blinded human-review sheet for cited Research Plan findings."""

import csv
import hashlib
import json
from pathlib import Path
import random

from benchmarks.plan_faithfulness import key_finding_claims


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "benchmarks" / "plan_faithfulness.json"
INPUT_CACHE = ROOT / ".academicforge_cache" / "plan_faithfulness" / "inputs.json"
OUT = ROOT / "docs" / "benchmarks" / "plan_entailment_blind.csv"
SAMPLE_SIZE = 20
SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_rows(artifact: dict, inputs: dict) -> list[dict]:
    papers_by_query = {item["id"]: item["papers"] for item in inputs["queries"]}
    rows = []
    for plan in artifact["plans"]:
        papers = papers_by_query[plan["id"]]
        findings = plan.get("key_findings") or key_finding_claims(plan["research_plan"])
        for index, finding in enumerate(findings, 1):
            cited = [papers[citation - 1] for citation in finding["citation_indices"] if 1 <= citation <= len(papers)]
            if not cited:
                continue
            evidence = "\n\n".join(
                f"[{citation}] {paper.get('title', 'Untitled')}\n{paper.get('abstract', '')}"
                for citation, paper in zip(finding["citation_indices"], cited)
            )
            rows.append(
                {
                    "item_id": plan["id"],
                    "finding_id": f"{plan['id']}-{index}",
                    "query": plan["query"],
                    "claim": finding["claim"],
                    "cited_evidence": evidence,
                    "citation_entailment_label": "",
                    "citation_notes": "",
                    "external_fact_check_label": "",
                    "external_source_urls": "",
                    "external_notes": "",
                }
            )
    return rows


def main():
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    inputs = json.loads(INPUT_CACHE.read_text(encoding="utf-8"))
    expected = artifact["inputs"]["frozen_input_sha256"]
    if sha256(INPUT_CACHE) != expected:
        raise ValueError("Frozen input cache does not match the Research Plan artifact")
    rows = review_rows(artifact, inputs)
    sample = random.Random(SEED).sample(rows, min(SAMPLE_SIZE, len(rows)))
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sample[0].keys())
        writer.writeheader()
        writer.writerows(sample)
    print(f"Wrote blinded 20-claim entailment review sheet: {OUT}")


if __name__ == "__main__":
    main()

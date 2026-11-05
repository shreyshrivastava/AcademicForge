"""Evaluate citation membership and Key Findings citation coverage in Research Plans.

The first run freezes retrieved paper inputs in the ignored cache. Later runs use
that snapshot, so live-search changes cannot alter the evaluation set.
"""

import argparse
from benchmarks.common import ROOT, write_json_atomically, file_sha256
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from backend.data_pipeline import retrieve_and_rank_papers
from backend.llm import model_name
from backend.research_plan_generator import CITATION_PATTERN, generate_research_plan, validate_evidence_citations
from benchmarks.faithfulness import DEFAULT_QUERIES, load_queries


DEFAULT_INPUT_CACHE = ROOT / ".academicforge_cache" / "plan_faithfulness" / "inputs.json"
DEFAULT_OUTPUT = ROOT / "docs" / "benchmarks" / "plan_faithfulness.json"
DEFAULT_PLAN_CACHE_DIR = ROOT / ".academicforge_cache" / "plan_faithfulness" / "plans"
DEFAULT_QUERY_LIMIT = 10
DEFAULT_PAPER_LIMIT = 3


def _key_finding_bullets(plan: str) -> list[str]:
    in_key_findings = False
    bullets = []
    for line in (plan or "").splitlines():
        stripped = line.strip()
        if stripped.lower() == "# key findings":
            in_key_findings = True
            continue
        if in_key_findings and stripped.startswith("#"):
            break
        if in_key_findings and stripped.startswith(("- ", "* ")):
            bullets.append(stripped)
    return bullets


def key_finding_claims(plan: str) -> list[dict]:
    """Extract reviewable Key Findings claims and their cited paper indices."""
    claims = []
    for bullet in _key_finding_bullets(plan):
        indices = [int(index) for index in CITATION_PATTERN.findall(bullet)]
        claims.append({"claim": CITATION_PATTERN.sub("", bullet[2:]).strip(), "citation_indices": indices})
    return claims


def key_findings_citation_coverage(plan: str) -> dict:
    """Count Key Findings bullets that end with a selected-paper citation."""
    bullets = _key_finding_bullets(plan)
    cited_bullets = sum(bool(CITATION_PATTERN.search(bullet)) for bullet in bullets)
    return {
        "bullets": len(bullets),
        "cited_bullets": cited_bullets,
        "coverage": cited_bullets / len(bullets) if bullets else 0.0,
    }


def paper_metadata(papers: list[dict]) -> list[dict]:
    return [{"paper_id": paper.get("paper_id"), "title": paper.get("title")} for paper in papers]


def plan_record(query_id: str, query: str, papers: list[dict], plan: str) -> dict:
    """Produce a tracked artifact record without copying source abstracts."""
    return {
        "id": query_id,
        "query": query,
        "papers": paper_metadata(papers),
        "research_plan": plan,
        "citation_membership": validate_evidence_citations(plan, papers),
        "key_findings_citation_coverage": key_findings_citation_coverage(plan),
        "key_findings": key_finding_claims(plan),
    }


def cached_plan_matches(payload: dict, query: str, papers: list[dict], model: str, temperature: float) -> bool:
    return (
        payload.get("query") == query
        and payload.get("paper_ids") == [paper.get("paper_id") for paper in papers]
        and payload.get("model") == model
        and payload.get("temperature") == temperature
        and isinstance(payload.get("research_plan"), str)
    )


def load_or_generate_plan(cache_dir: Path, item: dict, model: str, temperature: float) -> str:
    path = cache_dir / f"{item['id']}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if cached_plan_matches(payload, item["query"], item["papers"], model, temperature):
            print(f"[plan-faithfulness] Reusing {item['id']}", flush=True)
            return payload["research_plan"]
    plan = generate_research_plan(item["papers"], query=item["query"], model=model)
    write_json_atomically(
        path,
        {
            "query": item["query"],
            "paper_ids": [paper.get("paper_id") for paper in item["papers"]],
            "model": model,
            "temperature": temperature,
            "research_plan": plan,
        },
    )
    return plan


def prepare_inputs(queries: list[dict], paper_limit: int) -> dict:
    records = []
    for index, item in enumerate(queries, 1):
        print(f"[plan-faithfulness] Preparing {index}/{len(queries)}: {item['id']}", flush=True)
        papers = retrieve_and_rank_papers(item["query"])[:paper_limit]
        if not papers:
            raise RuntimeError(f"No papers were retrieved for {item['id']!r}")
        records.append({"id": item["id"], "query": item["query"], "papers": papers})
    return {"schema_version": 1, "queries": records}


def load_or_prepare_inputs(path: Path, queries: list[dict], paper_limit: int) -> dict:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or len(payload.get("queries", [])) != len(queries):
            raise ValueError(f"Frozen input cache is incompatible: {path}")
        return payload
    payload = prepare_inputs(queries, paper_limit)
    write_json_atomically(path, payload)
    return payload


def summarize(records: list[dict]) -> dict:
    total = len(records)
    membership_valid = sum(record["citation_membership"]["valid"] for record in records)
    with_cited_findings = sum(record["key_findings_citation_coverage"]["coverage"] == 1.0 for record in records)
    bullets = sum(record["key_findings_citation_coverage"]["bullets"] for record in records)
    cited_bullets = sum(record["key_findings_citation_coverage"]["cited_bullets"] for record in records)
    return {
        "plans": total,
        "plans_with_only_selected_paper_citations": membership_valid,
        "plans_with_all_key_findings_cited": with_cited_findings,
        "key_finding_bullets": bullets,
        "cited_key_finding_bullets": cited_bullets,
        "key_findings_citation_coverage": cited_bullets / bullets if bullets else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--input-cache", type=Path, default=DEFAULT_INPUT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-cache-dir", type=Path, default=DEFAULT_PLAN_CACHE_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT)
    parser.add_argument("--paper-limit", type=int, default=DEFAULT_PAPER_LIMIT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.limit <= 20:
        raise ValueError("--limit must be between 1 and 20")
    if args.paper_limit < 1:
        raise ValueError("--paper-limit must be at least 1")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("--temperature must be between 0 and 2")
    os.environ["LOCAL_LLM_TEMPERATURE"] = str(args.temperature)
    queries = load_queries(args.queries)[:args.limit]
    inputs = load_or_prepare_inputs(args.input_cache, queries, args.paper_limit)
    if args.prepare_only:
        print(f"Wrote frozen plan-evaluation inputs: {args.input_cache}")
        return

    model = model_name("research_plan")
    records = []
    for index, item in enumerate(inputs["queries"], 1):
        print(f"[plan-faithfulness] Generating {index}/{len(inputs['queries'])}: {item['id']}", flush=True)
        plan = load_or_generate_plan(args.plan_cache_dir, item, model, args.temperature)
        records.append(plan_record(item["id"], item["query"], item["papers"], plan))
    payload = {
        "benchmark": "AcademicForge Research Plan citation-faithfulness screen",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "query_fixture": str(args.queries.relative_to(ROOT)),
            "query_fixture_sha256": file_sha256(args.queries),
            "frozen_input_sha256": file_sha256(args.input_cache),
            "queries": len(records),
            "papers_per_query": args.paper_limit,
        },
        "model": model,
        "generation": {"temperature": args.temperature},
        "method": {
            "citation_membership": "Every plan citation index must refer to a selected paper.",
            "key_findings_coverage": "Every Key Findings bullet should contain a selected-paper citation.",
            "limitation": "These checks do not establish that a cited paper entails a claim; human or NLI review is still required.",
        },
        "summary": summarize(records),
        "plans": records,
    }
    write_json_atomically(args.output, payload)
    print(f"Wrote Research Plan citation-faithfulness artifact: {args.output}")


if __name__ == "__main__":
    main()

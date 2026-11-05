from benchmarks.plan_faithfulness import (
    cached_plan_matches,
    key_finding_claims,
    key_findings_citation_coverage,
    plan_record,
)
from scripts.build_plan_entailment_review import review_rows


PAPERS = [
    {"paper_id": "paper-1", "title": "First paper", "abstract": "Private source text."},
    {"paper_id": "paper-2", "title": "Second paper", "abstract": "Private source text."},
]


def test_key_findings_citation_coverage_counts_only_cited_bullets():
    plan = """# Research Plan
# Key Findings
- Supported finding. [1]
- Finding without a citation.
# Research Gaps
- A gap.
"""

    assert key_findings_citation_coverage(plan) == {
        "bullets": 2,
        "cited_bullets": 1,
        "coverage": 0.5,
    }


def test_key_finding_claims_keeps_only_key_finding_bullets_and_citations():
    plan = """# Research Plan
# Key Findings
- First supported finding. [1] [2]
* Second finding without a citation.
# Research Gaps
- Not a finding. [3]
"""

    assert key_finding_claims(plan) == [
        {"claim": "First supported finding.", "citation_indices": [1, 2]},
        {"claim": "Second finding without a citation.", "citation_indices": []},
    ]


def test_plan_record_excludes_abstracts_and_checks_selected_citations():
    record = plan_record(
        "q01",
        "example question",
        PAPERS,
        "# Research Plan\n# Key Findings\n- Supported. [1]\n# Research Gaps\n- Gap.",
    )

    assert record["citation_membership"]["valid"] is True
    assert record["key_findings_citation_coverage"]["coverage"] == 1.0
    assert record["papers"] == [
        {"paper_id": "paper-1", "title": "First paper"},
        {"paper_id": "paper-2", "title": "Second paper"},
    ]


def test_cached_plan_requires_the_same_query_papers_model_and_temperature():
    payload = {
        "query": "example question",
        "paper_ids": ["paper-1", "paper-2"],
        "model": "local-model",
        "temperature": 0.0,
        "research_plan": "A cached plan.",
    }

    assert cached_plan_matches(payload, "example question", PAPERS, "local-model", 0.0) is True
    assert cached_plan_matches(payload, "changed question", PAPERS, "local-model", 0.0) is False
    assert cached_plan_matches(payload, "example question", PAPERS, "local-model", 0.2) is False


def test_review_rows_exposes_only_cited_evidence_for_human_entailment_review():
    artifact = {
        "plans": [
            {
                "id": "q01",
                "query": "example question",
                "research_plan": "unused",
                "key_findings": [{"claim": "Supported claim.", "citation_indices": [2]}],
            }
        ]
    }
    inputs = {"queries": [{"id": "q01", "papers": PAPERS}]}

    rows = review_rows(artifact, inputs)

    assert rows[0]["claim"] == "Supported claim."
    assert "[2] Second paper" in rows[0]["cited_evidence"]
    assert "First paper" not in rows[0]["cited_evidence"]
    assert rows[0]["citation_entailment_label"] == ""
    assert rows[0]["external_source_urls"] == ""

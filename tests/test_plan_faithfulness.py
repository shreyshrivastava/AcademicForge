from benchmarks.plan_faithfulness import cached_plan_matches, key_findings_citation_coverage, plan_record


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


def test_cached_plan_requires_the_same_query_papers_and_model():
    payload = {
        "query": "example question",
        "paper_ids": ["paper-1", "paper-2"],
        "model": "local-model",
        "research_plan": "A cached plan.",
    }

    assert cached_plan_matches(payload, "example question", PAPERS, "local-model") is True
    assert cached_plan_matches(payload, "changed question", PAPERS, "local-model") is False

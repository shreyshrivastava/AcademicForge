from scripts.score_plan_entailment_review import summarize_rows


def test_summarize_rows_keeps_partial_support_separate_from_entailment():
    rows = [
        {"finding_id": "q01-1", "citation_entailment_label": "entailed", "external_fact_check_label": ""},
        {"finding_id": "q01-2", "citation_entailment_label": "partially_supported", "external_fact_check_label": "insufficient"},
        {"finding_id": "q01-3", "citation_entailment_label": "insufficient", "external_fact_check_label": "partially_supported"},
    ]

    summary = summarize_rows(rows)

    assert summary["claims_in_sheet"] == 3
    assert summary["citation_entailment"] == {
        "rated_claims": 3,
        "counts": {"entailed": 1, "partially_supported": 1, "insufficient": 1},
    }
    assert summary["external_fact_check"] == {
        "rated_claims": 2,
        "counts": {"partially_supported": 1, "insufficient": 1},
    }


def test_summarize_rows_rejects_duplicate_finding_ids():
    rows = [
        {"finding_id": "q01-1", "citation_entailment_label": "entailed", "external_fact_check_label": ""},
        {"finding_id": "q01-1", "citation_entailment_label": "insufficient", "external_fact_check_label": ""},
    ]

    try:
        summarize_rows(rows)
    except ValueError as error:
        assert "duplicate" in str(error).lower()
    else:
        raise AssertionError("Expected duplicate finding IDs to be rejected")

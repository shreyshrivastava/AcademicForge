from scripts.score_assisted_summary_review import summarize_rows


def test_summarize_rows_counts_decisions_without_treating_them_as_blind_ratings():
    rows = [
        {"item_id": "q01", "better_output": "A", "user_adjudication": "agrees", "output_a_faithfulness_1_to_5": "5", "output_b_faithfulness_1_to_5": "1", "output_a_usefulness_1_to_5": "4", "output_b_usefulness_1_to_5": "2"},
        {"item_id": "q02", "better_output": "B", "user_adjudication": "disagrees", "output_a_faithfulness_1_to_5": "", "output_b_faithfulness_1_to_5": "", "output_a_usefulness_1_to_5": "", "output_b_usefulness_1_to_5": ""},
    ]

    assert summarize_rows(rows) == {
        "items": 2,
        "better_output_counts": {"A": 1, "B": 1},
        "numeric_rating_items": 1,
        "adjudication_counts": {"agrees": 1, "disagrees": 1},
    }

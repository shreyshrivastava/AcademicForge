from benchmarks.faithfulness import score_sentence_embeddings, split_sentences


def test_split_sentences_ignores_extra_whitespace():
    assert split_sentences(" First claim.\n\nSecond claim! ") == ["First claim.", "Second claim!"]


def test_embedding_support_uses_best_abstract_sentence_and_threshold():
    records = score_sentence_embeddings(
        ["Supported.", "Unsupported."],
        ["Evidence one.", "Evidence two."],
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.9, 0.1], [0.4, 0.6]],
        threshold=0.8,
    )
    assert records[0]["supported"] is True
    assert records[0]["supporting_abstract_sentence"] == "Evidence one."
    assert records[1]["supported"] is False
    assert records[1]["max_similarity"] == 0.6

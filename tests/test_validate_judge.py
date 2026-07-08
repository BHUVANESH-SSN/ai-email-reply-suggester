from evaluator.calibration.validate_judge import compute_agreement, pearson_correlation


def test_pearson_correlation_perfect_positive():
    assert abs(pearson_correlation([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9


def test_pearson_correlation_perfect_negative():
    assert abs(pearson_correlation([1, 2, 3, 4], [4, 3, 2, 1]) - (-1.0)) < 1e-9


def test_pearson_correlation_no_variance_is_nan():
    result = pearson_correlation([3, 3, 3], [1, 2, 3])
    assert result != result  # NaN != NaN


def test_compute_agreement_matches_ids_and_computes_pct_within_one():
    eval_results = [
        {"id": "email-001", "judge": {"dimensions": {
            "relevance": {"score": 4}, "grounding": {"score": 5}, "tone": {"score": 4}, "completeness": {"score": 3},
        }}},
        {"id": "email-002", "judge": {"dimensions": {
            "relevance": {"score": 2}, "grounding": {"score": 5}, "tone": {"score": 3}, "completeness": {"score": 5},
        }}},
    ]
    human_labels = [
        {"id": "email-001", "human_scores": {"relevance": 4, "grounding": 5, "tone": 3, "completeness": 3}},
        {"id": "email-002", "human_scores": {"relevance": 3, "grounding": 5, "tone": 3, "completeness": 5}},
    ]
    agreement = compute_agreement(eval_results, human_labels)
    assert agreement["relevance"]["n"] == 2
    assert agreement["grounding"]["pct_within_1"] == 100.0
    assert agreement["completeness"]["pct_within_1"] == 100.0

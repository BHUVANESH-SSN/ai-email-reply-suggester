import pytest

from evaluator.rubric import DIMENSIONS, DIMENSION_KEYS, weighted_overall


def test_dimension_keys_match_expected_set():
    assert set(DIMENSION_KEYS) == {"relevance", "grounding", "tone", "completeness"}


def test_weights_sum_to_one():
    total = sum(d["weight"] for d in DIMENSIONS)
    assert abs(total - 1.0) < 1e-9


def test_weighted_overall_all_fives():
    result = weighted_overall({"relevance": 5, "grounding": 5, "tone": 5, "completeness": 5})
    assert result["scale_1_5"] == 5.0
    assert result["normalized_0_100"] == 100.0


def test_weighted_overall_all_ones():
    result = weighted_overall({"relevance": 1, "grounding": 1, "tone": 1, "completeness": 1})
    assert result["scale_1_5"] == 1.0
    assert result["normalized_0_100"] == 0.0


def test_weighted_overall_respects_weights():
    # relevance has the highest weight (0.40) - a low relevance score should
    # pull the overall down more than a low tone score (weight 0.15) would.
    low_relevance = weighted_overall({"relevance": 1, "grounding": 5, "tone": 5, "completeness": 5})
    low_tone = weighted_overall({"relevance": 5, "grounding": 5, "tone": 1, "completeness": 5})
    assert low_relevance["scale_1_5"] < low_tone["scale_1_5"]


def test_weighted_overall_raises_on_missing_dimension():
    with pytest.raises(ValueError, match="grounding"):
        weighted_overall({"relevance": 5, "tone": 5, "completeness": 5})

from unittest.mock import patch

from scripts.run_end_to_end import run_single

FAKE_GEN_RESULT = {"reply": "Hi, sure thing. Best", "retrieved": [{"id": "email-001", "similarity": 0.8}], "model": "fake-gen-model"}
FAKE_EVAL_RESULT = {
    "id": "ad-hoc",
    "category": "unknown",
    "tone": "unknown",
    "generated_reply": "Hi, sure thing. Best",
    "judge": {"dimensions": {}, "overall": {"scale_1_5": 4.0, "normalized_0_100": 75.0}, "model": "fake-judge-model"},
    "heuristics": {},
    "supporting_metrics": {},
}


def test_run_single_wires_generation_and_evaluation():
    dataset_rows = [{"id": "email-001", "category": "sales", "sent_reply": "x" * 50}]

    with patch("scripts.run_end_to_end.generate_reply", return_value=FAKE_GEN_RESULT) as gen_mock, \
         patch("scripts.run_end_to_end.evaluate_response", return_value=FAKE_EVAL_RESULT) as eval_mock:
        result = run_single("Subject", "Body", dataset_rows)

    gen_mock.assert_called_once_with("Subject", "Body", dataset_rows)
    eval_mock.assert_called_once()
    assert result["retrieved_examples"] == FAKE_GEN_RESULT["retrieved"]
    assert result["judge"]["overall"]["normalized_0_100"] == 75.0

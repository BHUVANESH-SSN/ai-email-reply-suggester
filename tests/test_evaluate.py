from unittest.mock import patch

from evaluator.evaluate import aggregate_scores, evaluate_dataset, evaluate_response


FAKE_JUDGE_RESULT = {
    "dimensions": {
        "relevance": {"score": 4, "justification": "x"},
        "grounding": {"score": 5, "justification": "x"},
        "tone": {"score": 4, "justification": "x"},
        "completeness": {"score": 4, "justification": "x"},
    },
    "overall": {"scale_1_5": 4.25, "normalized_0_100": 81.25},
    "model": "fake-judge-model",
}


def test_evaluate_response_assembles_full_result():
    row = {
        "id": "email-001",
        "incoming_email": {"subject": "Order ORD999 status", "body": "Any update on ORD999?"},
        "sent_reply": "Hi, ORD999 shipped today. Best, Support",
        "category": "customer_support",
        "tone": "formal",
    }
    generated_reply = "Hi, thanks for checking in on ORD999 — it shipped today. Best, Support"

    with patch("evaluator.evaluate.judge_reply", return_value=FAKE_JUDGE_RESULT), \
         patch("evaluator.evaluate.embedding_similarity", return_value=0.93):
        result = evaluate_response(row, generated_reply, category_medians={"customer_support": 60})

    assert result["id"] == "email-001"
    assert result["judge"] == FAKE_JUDGE_RESULT
    assert result["heuristics"]["has_greeting"] is True
    assert result["heuristics"]["has_signoff"] is True
    assert result["heuristics"]["entity_echo_score"] == 1.0
    assert result["supporting_metrics"]["sent_reply_similarity"] == 0.93


def test_evaluate_response_omits_similarity_without_sent_reply():
    row = {
        "id": "ad-hoc",
        "incoming_email": {"subject": "s", "body": "b"},
        "sent_reply": "",
        "category": "unknown",
        "tone": "unknown",
    }
    with patch("evaluator.evaluate.judge_reply", return_value=FAKE_JUDGE_RESULT):
        result = evaluate_response(row, "some reply", category_medians={})
    assert "sent_reply_similarity" not in result["supporting_metrics"]


def test_evaluate_dataset_excludes_current_row_from_retrieval_pool():
    dataset_rows = [
        {"id": "email-001", "incoming_email": {"subject": "s1", "body": "b1"}, "sent_reply": "r1", "category": "sales", "tone": "formal"},
        {"id": "email-002", "incoming_email": {"subject": "s2", "body": "b2"}, "sent_reply": "r2", "category": "sales", "tone": "formal"},
    ]
    seen_pools = []

    def fake_generate_fn(subject, body, retrieval_pool):
        seen_pools.append([r["id"] for r in retrieval_pool])
        return {"reply": "generated"}

    with patch("evaluator.evaluate.judge_reply", return_value=FAKE_JUDGE_RESULT):
        evaluate_dataset(dataset_rows, fake_generate_fn)

    assert seen_pools == [["email-002"], ["email-001"]]


def test_evaluate_dataset_uses_separate_retrieval_corpus_when_given():
    rows_to_evaluate = [
        {"id": "email-001", "incoming_email": {"subject": "s1", "body": "b1"}, "sent_reply": "r1", "category": "sales", "tone": "formal"},
    ]
    retrieval_corpus = [
        {"id": "email-001", "incoming_email": {"subject": "s1", "body": "b1"}, "sent_reply": "r1", "category": "sales", "tone": "formal"},
        {"id": "email-002", "incoming_email": {"subject": "s2", "body": "b2"}, "sent_reply": "r2", "category": "sales", "tone": "formal"},
    ]
    seen_pools = []

    def fake_generate_fn(subject, body, retrieval_pool):
        seen_pools.append([r["id"] for r in retrieval_pool])
        return {"reply": "generated"}

    with patch("evaluator.evaluate.judge_reply", return_value=FAKE_JUDGE_RESULT):
        evaluate_dataset(rows_to_evaluate, fake_generate_fn, retrieval_corpus=retrieval_corpus)

    assert seen_pools == [["email-002"]]


def test_aggregate_scores_computes_stats_and_breakdowns():
    results = [
        {"category": "sales", "tone": "formal", "judge": {"overall": {"normalized_0_100": 80.0}}},
        {"category": "sales", "tone": "casual", "judge": {"overall": {"normalized_0_100": 60.0}}},
        {"category": "internal", "tone": "casual", "judge": {"overall": {"normalized_0_100": 100.0}}},
    ]
    summary = aggregate_scores(results)
    assert summary["count"] == 3
    assert summary["errors"] == 0
    assert summary["mean"] == 80.0
    assert summary["min"] == 60.0
    assert summary["max"] == 100.0
    assert summary["by_category"]["sales"] == 70.0
    assert summary["by_category"]["internal"] == 100.0
    assert summary["by_tone"]["casual"] == 80.0


def test_aggregate_scores_excludes_error_rows():
    results = [
        {"category": "sales", "tone": "formal", "judge": {"overall": {"normalized_0_100": 80.0}}},
        {"id": "email-002", "category": "sales", "tone": "casual", "error": "boom"},
    ]
    summary = aggregate_scores(results)
    assert summary["count"] == 1
    assert summary["errors"] == 1
    assert summary["mean"] == 80.0


def test_evaluate_dataset_records_error_and_continues_on_row_failure():
    dataset_rows = [
        {"id": "email-001", "incoming_email": {"subject": "s1", "body": "b1"}, "sent_reply": "r1", "category": "sales", "tone": "formal"},
        {"id": "email-002", "incoming_email": {"subject": "s2", "body": "b2"}, "sent_reply": "r2", "category": "sales", "tone": "formal"},
    ]

    def flaky_generate_fn(subject, body, retrieval_pool):
        if subject == "s1":
            raise RuntimeError("simulated API failure")
        return {"reply": "generated"}

    with patch("evaluator.evaluate.judge_reply", return_value=FAKE_JUDGE_RESULT):
        results = evaluate_dataset(dataset_rows, flaky_generate_fn)

    assert len(results) == 2
    assert results[0]["id"] == "email-001"
    assert results[0]["error"] == "simulated API failure"
    assert results[1]["id"] == "email-002"
    assert "error" not in results[1]
    assert results[1]["judge"] == FAKE_JUDGE_RESULT


def test_evaluate_dataset_calls_on_result_for_each_row():
    dataset_rows = [
        {"id": "email-001", "incoming_email": {"subject": "s1", "body": "b1"}, "sent_reply": "r1", "category": "sales", "tone": "formal"},
        {"id": "email-002", "incoming_email": {"subject": "s2", "body": "b2"}, "sent_reply": "r2", "category": "sales", "tone": "formal"},
    ]
    seen = []

    def fake_generate_fn(subject, body, retrieval_pool):
        return {"reply": "generated"}

    with patch("evaluator.evaluate.judge_reply", return_value=FAKE_JUDGE_RESULT):
        results = evaluate_dataset(dataset_rows, fake_generate_fn, on_result=seen.append)

    assert [r["id"] for r in seen] == ["email-001", "email-002"]
    assert seen == results

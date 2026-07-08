import json

import numpy as np

from generator.retrieve import load_dataset, top_k_similar


def test_top_k_similar_orders_by_cosine_similarity():
    # 4 unit vectors in 2D; query is closest to index 2, then 0, then 3, then 1
    query = np.array([1.0, 0.0])
    corpus = np.array(
        [
            [0.7, 0.7],   # index 0: ~45 degrees off
            [0.0, 1.0],   # index 1: 90 degrees off (least similar)
            [0.99, 0.14], # index 2: nearly identical (most similar)
            [0.5, 0.87],  # index 3: ~60 degrees off
        ]
    )
    result = top_k_similar(query, corpus, k=2)
    ordered_indices = [idx for idx, _ in result]
    assert ordered_indices == [2, 0]


def test_top_k_similar_returns_float_similarities():
    query = np.array([1.0, 0.0])
    corpus = np.array([[1.0, 0.0], [0.0, 1.0]])
    result = top_k_similar(query, corpus, k=1)
    assert result[0][0] == 0
    assert abs(result[0][1] - 1.0) < 1e-6


def test_load_dataset_reads_jsonl(tmp_path):
    path = tmp_path / "emails.jsonl"
    rows = [
        {"id": "email-001", "incoming_email": {"subject": "s", "body": "b"}, "sent_reply": "r", "category": "sales", "tone": "formal"},
        {"id": "email-002", "incoming_email": {"subject": "s2", "body": "b2"}, "sent_reply": "r2", "category": "internal", "tone": "casual"},
    ]
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    loaded = load_dataset(str(path))
    assert len(loaded) == 2
    assert loaded[0]["id"] == "email-001"
    assert loaded[1]["category"] == "internal"

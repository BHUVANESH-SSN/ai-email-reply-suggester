import json
from unittest.mock import MagicMock, patch

import pytest

from evaluator.judge import build_judge_messages, judge_reply, parse_judge_response

VALID_JUDGE_JSON = json.dumps(
    {
        "relevance": {"score": 4, "justification": "Addresses the main ask."},
        "grounding": {"score": 5, "justification": "No invented facts."},
        "tone": {"score": 4, "justification": "Matches formal register."},
        "completeness": {"score": 3, "justification": "Misses the second question."},
    }
)


def test_parse_judge_response_valid():
    parsed = parse_judge_response(VALID_JUDGE_JSON)
    assert parsed["relevance"]["score"] == 4
    assert "justification" in parsed["completeness"]


def test_parse_judge_response_missing_dimension_raises():
    bad = json.dumps(
        {
            "relevance": {"score": 4, "justification": "x"},
            "grounding": {"score": 5, "justification": "x"},
            "tone": {"score": 4, "justification": "x"},
        }
    )
    with pytest.raises(ValueError, match="completeness"):
        parse_judge_response(bad)


def test_parse_judge_response_invalid_score_raises():
    bad = json.dumps(
        {
            "relevance": {"score": 9, "justification": "x"},
            "grounding": {"score": 5, "justification": "x"},
            "tone": {"score": 4, "justification": "x"},
            "completeness": {"score": 3, "justification": "x"},
        }
    )
    with pytest.raises(ValueError, match="relevance"):
        parse_judge_response(bad)


def test_build_judge_messages_includes_sent_reply_when_given():
    messages = build_judge_messages("Subj", "Body", "Generated reply text", sent_reply="Actual sent reply")
    joined = " ".join(m["content"] for m in messages)
    assert "Generated reply text" in joined
    assert "Actual sent reply" in joined


def test_build_judge_messages_omits_sent_reply_when_absent():
    messages = build_judge_messages("Subj", "Body", "Generated reply text")
    joined = " ".join(m["content"] for m in messages)
    assert "Generated reply text" in joined
    assert "actually sent" not in joined.lower()


def test_judge_reply_wires_completion_to_rubric():
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(content=VALID_JUDGE_JSON))]

    with patch("evaluator.judge.create_chat_completion", return_value=fake_completion):
        result = judge_reply("Subj", "Body", "Generated reply")

    assert result["dimensions"]["relevance"]["score"] == 4
    assert 0 <= result["overall"]["normalized_0_100"] <= 100
    assert "model" in result

import json

import pytest

from dataset.build_dataset import (
    CATEGORY_TONES,
    build_generation_prompt,
    parse_generated_pairs,
)


def test_category_tones_covers_required_categories():
    categories = {c for c, _ in CATEGORY_TONES}
    assert categories == {
        "customer_support",
        "scheduling",
        "sales",
        "complaint",
        "internal",
    }


def test_build_generation_prompt_includes_category_tone_and_count():
    prompt = build_generation_prompt("scheduling", "casual", n=5)
    assert "scheduling" in prompt
    assert "casual" in prompt
    assert "5" in prompt


def test_parse_generated_pairs_plain_json():
    raw = json.dumps(
        [
            {
                "incoming_subject": "Reschedule?",
                "incoming_body": "Can we move our call?",
                "sent_reply": "Sure, how about Thursday?",
            }
        ]
    )
    pairs = parse_generated_pairs(raw)
    assert len(pairs) == 1
    assert pairs[0]["incoming_subject"] == "Reschedule?"


def test_parse_generated_pairs_handles_markdown_fence():
    raw = (
        "```json\n"
        + json.dumps(
            [
                {
                    "incoming_subject": "s",
                    "incoming_body": "b",
                    "sent_reply": "r",
                }
            ]
        )
        + "\n```"
    )
    pairs = parse_generated_pairs(raw)
    assert len(pairs) == 1


def test_parse_generated_pairs_raises_on_missing_key():
    raw = json.dumps([{"incoming_subject": "s", "incoming_body": "b"}])
    with pytest.raises(ValueError, match="sent_reply"):
        parse_generated_pairs(raw)

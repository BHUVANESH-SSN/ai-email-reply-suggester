from generator.prompt_templates import build_few_shot_block, build_messages


FIXTURE_RETRIEVED = [
    {
        "row": {
            "id": "email-001",
            "incoming_email": {"subject": "Can we move Tuesday's call?", "body": "Something came up."},
            "sent_reply": "No problem, how about Wednesday at 2pm?",
        },
        "similarity": 0.87,
    }
]


def test_build_few_shot_block_includes_example_content():
    block = build_few_shot_block(FIXTURE_RETRIEVED)
    assert "Can we move Tuesday's call?" in block
    assert "No problem, how about Wednesday at 2pm?" in block


def test_build_messages_has_system_and_user_roles():
    messages = build_messages("Reschedule request", "Can we push to Friday?", FIXTURE_RETRIEVED)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Reschedule request" in messages[1]["content"]
    assert "Can we push to Friday?" in messages[1]["content"]
    assert "Wednesday at 2pm" in messages[1]["content"]

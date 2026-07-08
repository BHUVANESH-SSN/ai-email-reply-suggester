from unittest.mock import MagicMock, patch

from generator.generate_reply import generate_reply


def test_generate_reply_wires_retrieval_and_completion():
    fake_retrieved = [
        {
            "row": {"id": "email-001", "incoming_email": {"subject": "s", "body": "b"}, "sent_reply": "r"},
            "similarity": 0.91,
        }
    ]
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=MagicMock(content="Hi,\nThanks for reaching out.\nBest"))]

    with patch("generator.generate_reply.retrieve_similar", return_value=fake_retrieved), \
         patch("generator.generate_reply.create_chat_completion", return_value=fake_completion) as fake_create:
        result = generate_reply("New subject", "New body", dataset_rows=[], k=1)

    assert result["reply"] == "Hi,\nThanks for reaching out.\nBest"
    assert result["retrieved"] == [{"id": "email-001", "similarity": 0.91}]
    assert "model" in result
    fake_create.assert_called_once()

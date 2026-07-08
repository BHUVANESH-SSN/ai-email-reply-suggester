from evaluator.heuristics import (
    category_median_lengths,
    entity_echo_score,
    extract_entities,
    has_greeting,
    has_signoff,
    length_ratio,
)


def test_has_greeting_true_and_false():
    assert has_greeting("Hi John,\nThanks for your note.") is True
    assert has_greeting("The order shipped yesterday.") is False


def test_has_signoff_true_and_false():
    assert has_signoff("Sounds good.\nBest,\nJane") is True
    assert has_signoff("Sounds good. See you then.") is False


def test_extract_entities_finds_names_dates_and_order_numbers():
    text = "Hi, this is John Smith. Your order ORD12345 shipped on 5/12/2026."
    entities = extract_entities(text)
    assert "John Smith" in entities
    assert "ORD12345" in entities
    assert "5/12/2026" in entities


def test_entity_echo_score_full_and_zero():
    incoming = "Hi, this is John Smith regarding order ORD12345."
    good_reply = "Hi John Smith, thanks for asking about ORD12345 — it shipped today."
    bad_reply = "Thanks for your message, we'll get back to you soon."
    assert entity_echo_score(incoming, good_reply) == 1.0
    assert entity_echo_score(incoming, bad_reply) == 0.0


def test_entity_echo_score_no_entities_returns_one():
    assert entity_echo_score("hello there", "hi") == 1.0


def test_category_median_lengths():
    rows = [
        {"category": "sales", "sent_reply": "a" * 100},
        {"category": "sales", "sent_reply": "a" * 200},
        {"category": "internal", "sent_reply": "a" * 50},
    ]
    medians = category_median_lengths(rows)
    assert medians["sales"] == 150
    assert medians["internal"] == 50


def test_length_ratio():
    assert length_ratio("a" * 100, 100) == 1.0
    assert length_ratio("a" * 50, 100) == 0.5
    assert length_ratio("a" * 10, 0) == 1.0

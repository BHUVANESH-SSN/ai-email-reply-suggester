import re
import statistics

GREETING_PATTERNS = [
    r"\bhi\b",
    r"\bhello\b",
    r"\bdear\b",
    r"\bgood (morning|afternoon|evening)\b",
]
SIGNOFF_PATTERNS = [
    r"\bbest\b",
    r"\bregards\b",
    r"\bthanks\b",
    r"\bthank you\b",
    r"\bsincerely\b",
    r"\bcheers\b",
]

ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+\s[A-Z][a-z]+|\d{1,2}/\d{1,2}(?:/\d{2,4})?|[A-Z]{2,}\d{2,}|\d{2,}[A-Z]{2,}\d*)\b"
)


def has_greeting(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in GREETING_PATTERNS)


def has_signoff(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in SIGNOFF_PATTERNS)


def extract_entities(text: str) -> set:
    return set(ENTITY_PATTERN.findall(text))


def entity_echo_score(incoming_text: str, reply_text: str) -> float:
    incoming_entities = extract_entities(incoming_text)
    if not incoming_entities:
        return 1.0
    reply_lower = reply_text.lower()
    echoed = [e for e in incoming_entities if e.lower() in reply_lower]
    return len(echoed) / len(incoming_entities)


def category_median_lengths(dataset_rows: list) -> dict:
    lengths_by_category = {}
    for row in dataset_rows:
        lengths_by_category.setdefault(row["category"], []).append(len(row["sent_reply"]))
    return {cat: statistics.median(lens) for cat, lens in lengths_by_category.items()}


def length_ratio(reply_text: str, category_median_length: float) -> float:
    if not category_median_length:
        return 1.0
    return len(reply_text) / category_median_length

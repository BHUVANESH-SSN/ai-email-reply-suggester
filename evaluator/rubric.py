DIMENSIONS = [
    {
        "key": "relevance",
        "name": "Relevance / task completion",
        "weight": 0.40,
        "description": "Does the reply actually address what the incoming email asked for?",
    },
    {
        "key": "grounding",
        "name": "Factual grounding / no hallucination",
        "weight": 0.25,
        "description": "Does the reply invent facts, commitments, dates, or details not present in the incoming email?",
    },
    {
        "key": "tone",
        "name": "Tone/style fit",
        "weight": 0.15,
        "description": "Does the reply match the expected tone (formal/casual/apologetic) for this category?",
    },
    {
        "key": "completeness",
        "name": "Completeness",
        "weight": 0.20,
        "description": "Does it cover all distinct asks if the incoming email has multiple questions?",
    },
]

DIMENSION_KEYS = [d["key"] for d in DIMENSIONS]

_WEIGHT_BY_KEY = {d["key"]: d["weight"] for d in DIMENSIONS}


def weighted_overall(scores: dict) -> dict:
    missing = [k for k in DIMENSION_KEYS if k not in scores]
    if missing:
        raise ValueError(f"Missing dimension scores: {missing}")
    total_weight = sum(_WEIGHT_BY_KEY.values())
    weighted_sum = sum(scores[k] * _WEIGHT_BY_KEY[k] for k in DIMENSION_KEYS)
    scale_score = weighted_sum / total_weight
    return {
        "scale_1_5": round(scale_score, 3),
        "normalized_0_100": round((scale_score - 1) / 4 * 100, 1),
    }

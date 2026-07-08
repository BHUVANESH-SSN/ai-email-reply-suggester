import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from generator.config import GEN_MODEL, get_groq_client

CATEGORY_TONES = [
    ("customer_support", "formal"),
    ("customer_support", "apologetic"),
    ("scheduling", "casual"),
    ("scheduling", "formal"),
    ("sales", "formal"),
    ("sales", "casual"),
    ("complaint", "apologetic"),
    ("complaint", "formal"),
    ("internal", "casual"),
]

PAIRS_PER_COMBO = 5  # 9 combos * 5 = 45 pairs

GENERATION_PROMPT = """Generate {n} realistic, distinct example pairs of (incoming email, sent reply) for the category "{category}" with a "{tone}" tone.

Each example must be a plausible business email scenario for that category, and the reply must be the kind of reply a real person would actually send.

Return ONLY a JSON array, no prose, no markdown fence, in this exact schema:
[
  {{
    "incoming_subject": "...",
    "incoming_body": "...",
    "sent_reply": "..."
  }}
]

Make each of the {n} examples cover a clearly different scenario within the category (different names, different specific asks). Do not reuse the same scenario twice."""


def build_generation_prompt(category: str, tone: str, n: int = PAIRS_PER_COMBO) -> str:
    return GENERATION_PROMPT.format(n=n, category=category, tone=tone)


def parse_generated_pairs(raw_text: str) -> list:
    text = raw_text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in generation output: {text[:200]!r}")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("Parsed JSON is not a list")
    required_keys = ("incoming_subject", "incoming_body", "sent_reply")
    for item in data:
        for key in required_keys:
            if key not in item:
                raise ValueError(f"Missing key {key!r} in generated pair: {item}")
    return data


def generate_dataset(pairs_per_combo: int = PAIRS_PER_COMBO) -> list:
    client = get_groq_client()
    rows = []
    row_id = 1
    for category, tone in CATEGORY_TONES:
        prompt = build_generation_prompt(category, tone, pairs_per_combo)
        response = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )
        pairs = parse_generated_pairs(response.choices[0].message.content)
        for pair in pairs:
            rows.append(
                {
                    "id": f"email-{row_id:03d}",
                    "incoming_email": {
                        "subject": pair["incoming_subject"],
                        "body": pair["incoming_body"],
                    },
                    "sent_reply": pair["sent_reply"],
                    "category": category,
                    "tone": tone,
                    "metadata": {
                        "source": "synthetic-groq",
                        "model": GEN_MODEL,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )
            row_id += 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset/emails.jsonl")
    parser.add_argument("--pairs-per-combo", type=int, default=PAIRS_PER_COMBO)
    args = parser.parse_args()

    rows = generate_dataset(args.pairs_per_combo)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} pairs to {out_path}")


if __name__ == "__main__":
    main()

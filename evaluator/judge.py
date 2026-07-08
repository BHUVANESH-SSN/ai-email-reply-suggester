import json
import re

from generator.config import JUDGE_MODEL, create_chat_completion
from evaluator.rubric import DIMENSION_KEYS, DIMENSIONS, weighted_overall

_DIMENSION_LINES = "\n".join(f"- {d['key']}: {d['name']} — {d['description']}" for d in DIMENSIONS)

JUDGE_SYSTEM_PROMPT = f"""You are an impartial evaluator scoring a suggested email reply.
Score the reply on each dimension below, from 1 (very poor) to 5 (excellent).
Return ONLY a JSON object, no prose, no markdown fence, with this exact schema:
{{
  "<dimension_key>": {{"score": <1-5 integer>, "justification": "<one sentence>"}},
  ...
}}

Dimensions:
{_DIMENSION_LINES}"""


def build_judge_messages(subject: str, body: str, generated_reply: str, sent_reply: str = None) -> list:
    context = (
        f"Incoming email:\nSubject: {subject}\n{body}\n\n"
        f"Suggested reply to evaluate:\n{generated_reply}\n"
    )
    if sent_reply:
        context += (
            "\nFor reference, here is a reply that was actually sent to a similar email in the "
            f"past (one valid answer, not the only correct one):\n{sent_reply}\n"
        )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]


def parse_judge_response(raw_text: str) -> dict:
    text = raw_text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {text[:200]!r}")
    data = json.loads(match.group(0))
    missing = [k for k in DIMENSION_KEYS if k not in data]
    if missing:
        raise ValueError(f"Judge response missing dimensions: {missing}")
    for key in DIMENSION_KEYS:
        score = data[key].get("score")
        if not isinstance(score, int) or not (1 <= score <= 5):
            raise ValueError(f"Invalid score for {key}: {score!r}")
    return data


def judge_reply(subject: str, body: str, generated_reply: str, sent_reply: str = None) -> dict:
    messages = build_judge_messages(subject, body, generated_reply, sent_reply)
    response = create_chat_completion(
        model=JUDGE_MODEL,
        messages=messages,
        temperature=0.0,
    )
    dimension_scores = parse_judge_response(response.choices[0].message.content)
    scores_only = {k: dimension_scores[k]["score"] for k in DIMENSION_KEYS}
    overall = weighted_overall(scores_only)
    return {
        "dimensions": dimension_scores,
        "overall": overall,
        "model": JUDGE_MODEL,
    }

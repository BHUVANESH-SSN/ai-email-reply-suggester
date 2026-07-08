import json

from evaluator.judge import judge_reply

ADVERSARIAL_CASES = [
    {
        "label": "bad_wrong_topic",
        "subject": "Refund status?",
        "body": "I returned item #4521 two weeks ago and haven't heard back.",
        "reply": "Hi! Thanks for your interest in our enterprise pricing plans. Our sales team will reach out to schedule a demo shortly. Best, Sales",
    },
    {
        "label": "bad_hallucinated_commitment",
        "subject": "Refund status?",
        "body": "I returned item #4521 two weeks ago and haven't heard back.",
        "reply": "Hi, your refund of $412.50 was processed on March 3rd and a replacement unit ships free of charge tomorrow via overnight courier. Best, Support",
    },
    {
        "label": "bad_rude_tone",
        "subject": "Refund status?",
        "body": "I returned item #4521 two weeks ago and haven't heard back.",
        "reply": "We get a lot of these requests. You'll hear back whenever we get to it.",
    },
    {
        "label": "good_direct_answer",
        "subject": "Refund status?",
        "body": "I returned item #4521 two weeks ago and haven't heard back.",
        "reply": "Hi, thanks for your patience — I checked and item #4521 was received by our warehouse and your refund is being processed now; you should see it within 3-5 business days. Best, Support",
    },
    {
        "label": "good_paraphrase",
        "subject": "Refund status?",
        "body": "I returned item #4521 two weeks ago and haven't heard back.",
        "reply": "Hello, apologies for the delay. We've confirmed receipt of item #4521 and your refund is in progress, expected within 3-5 business days. Thanks for bearing with us.",
    },
]


def run_adversarial_check() -> dict:
    results = {}
    for case in ADVERSARIAL_CASES:
        judged = judge_reply(case["subject"], case["body"], case["reply"])
        results[case["label"]] = judged["overall"]["normalized_0_100"]
    bad_scores = [v for k, v in results.items() if k.startswith("bad_")]
    good_scores = [v for k, v in results.items() if k.startswith("good_")]
    separation_ok = max(bad_scores) < min(good_scores)
    return {"scores": results, "separation_ok": separation_ok}


def run_consistency_check(subject: str, body: str, generated_reply: str) -> dict:
    first = judge_reply(subject, body, generated_reply)
    second = judge_reply(subject, body, generated_reply)
    diffs = {
        key: abs(first["dimensions"][key]["score"] - second["dimensions"][key]["score"])
        for key in first["dimensions"]
    }
    stable = all(diff <= 1 for diff in diffs.values())
    return {"first": first["overall"], "second": second["overall"], "diffs": diffs, "stable": stable}


if __name__ == "__main__":
    adversarial = run_adversarial_check()
    print("Adversarial check:")
    print(json.dumps(adversarial, indent=2))

    consistency = run_consistency_check(
        "Refund status?",
        "I returned item #4521 two weeks ago and haven't heard back.",
        "Hi, thanks for your patience — item #4521 was received and your refund is processing, expected within 3-5 business days. Best, Support",
    )
    print("\nConsistency check:")
    print(json.dumps(consistency, indent=2))

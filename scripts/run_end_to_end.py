import argparse
import json

from evaluator.evaluate import aggregate_scores, evaluate_dataset, evaluate_response
from evaluator.heuristics import category_median_lengths
from generator.generate_reply import generate_reply
from generator.retrieve import load_dataset


def run_single(subject: str, body: str, dataset_rows: list) -> dict:
    gen_result = generate_reply(subject, body, dataset_rows)
    category_medians = category_median_lengths(dataset_rows)
    fake_row = {
        "id": "ad-hoc",
        "incoming_email": {"subject": subject, "body": body},
        "sent_reply": "",
        "category": "unknown",
        "tone": "unknown",
    }
    eval_result = evaluate_response(fake_row, gen_result["reply"], category_medians)
    eval_result["retrieved_examples"] = gen_result["retrieved"]
    return eval_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject")
    parser.add_argument("--body")
    parser.add_argument("--dataset", default="dataset/emails.jsonl")
    parser.add_argument("--full-dataset", action="store_true")
    args = parser.parse_args()

    dataset_rows = load_dataset(args.dataset)

    if args.full_dataset:
        results = evaluate_dataset(dataset_rows, lambda subject, body, rows: generate_reply(subject, body, rows))
        print(json.dumps(aggregate_scores(results), indent=2))
    else:
        if not args.subject or not args.body:
            raise SystemExit("--subject and --body are required unless --full-dataset is set")
        result = run_single(args.subject, args.body, dataset_rows)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

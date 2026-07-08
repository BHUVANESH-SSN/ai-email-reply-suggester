import argparse
import json
import statistics
import sys
from pathlib import Path

from evaluator.heuristics import (
    category_median_lengths,
    entity_echo_score,
    has_greeting,
    has_signoff,
    length_ratio,
)
from evaluator.judge import judge_reply
from generator.retrieve import embed_texts, load_dataset


def embedding_similarity(text_a: str, text_b: str) -> float:
    vectors = embed_texts([text_a, text_b])
    return float(vectors[0] @ vectors[1])


def evaluate_response(row: dict, generated_reply: str, category_medians: dict) -> dict:
    incoming_text = f"{row['incoming_email']['subject']}\n{row['incoming_email']['body']}"
    judge_result = judge_reply(
        row["incoming_email"]["subject"],
        row["incoming_email"]["body"],
        generated_reply,
        sent_reply=row.get("sent_reply") or None,
    )
    median_len = category_medians.get(row["category"], 0)
    heuristics_result = {
        "has_greeting": has_greeting(generated_reply),
        "has_signoff": has_signoff(generated_reply),
        "entity_echo_score": entity_echo_score(incoming_text, generated_reply),
        "length_ratio": length_ratio(generated_reply, median_len),
    }
    supporting_metrics = {}
    if row.get("sent_reply"):
        supporting_metrics["sent_reply_similarity"] = embedding_similarity(
            generated_reply, row["sent_reply"]
        )
    return {
        "id": row["id"],
        "category": row["category"],
        "tone": row["tone"],
        "generated_reply": generated_reply,
        "judge": judge_result,
        "heuristics": heuristics_result,
        "supporting_metrics": supporting_metrics,
    }


def evaluate_dataset(rows_to_evaluate: list, generate_fn, retrieval_corpus: list = None, on_result=None) -> list:
    corpus = retrieval_corpus if retrieval_corpus is not None else rows_to_evaluate
    category_medians = category_median_lengths(corpus)
    results = []
    for row in rows_to_evaluate:
        retrieval_pool = [r for r in corpus if r["id"] != row["id"]]
        try:
            gen_result = generate_fn(row["incoming_email"]["subject"], row["incoming_email"]["body"], retrieval_pool)
            result = evaluate_response(row, gen_result["reply"], category_medians)
        except Exception as exc:
            result = {"id": row["id"], "category": row["category"], "tone": row["tone"], "error": str(exc)}
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def aggregate_scores(results: list) -> dict:
    scored = [r for r in results if "error" not in r]
    overalls = [r["judge"]["overall"]["normalized_0_100"] for r in scored]
    summary = {
        "count": len(scored),
        "errors": len(results) - len(scored),
        "mean": round(statistics.mean(overalls), 2),
        "min": round(min(overalls), 2),
        "max": round(max(overalls), 2),
        "stdev": round(statistics.stdev(overalls), 2) if len(overalls) > 1 else 0.0,
    }
    by_category = {}
    for r in scored:
        by_category.setdefault(r["category"], []).append(r["judge"]["overall"]["normalized_0_100"])
    summary["by_category"] = {cat: round(statistics.mean(vals), 2) for cat, vals in by_category.items()}
    by_tone = {}
    for r in scored:
        by_tone.setdefault(r["tone"], []).append(r["judge"]["overall"]["normalized_0_100"])
    summary["by_tone"] = {tone: round(statistics.mean(vals), 2) for tone, vals in by_tone.items()}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/emails.jsonl")
    parser.add_argument("--out", default=None, help="defaults to results/eval_results.jsonl for full-dataset runs; single --id runs are not written to disk unless --out is given explicitly")
    parser.add_argument("--id", help="evaluate only this single dataset row id")
    args = parser.parse_args()

    from generator.generate_reply import generate_reply

    def generate_fn(subject, body, dataset_rows):
        return generate_reply(subject, body, dataset_rows)

    dataset_rows = load_dataset(args.dataset)
    rows_to_evaluate = dataset_rows
    if args.id:
        rows_to_evaluate = [r for r in dataset_rows if r["id"] == args.id]
        if not rows_to_evaluate:
            raise SystemExit(f"No row with id {args.id!r} found in {args.dataset}")

    out_arg = args.out
    if out_arg is None and not args.id:
        out_arg = "results/eval_results.jsonl"

    out_file = None
    if out_arg is not None:
        out_path = Path(out_arg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path.open("w")

    def on_result(result):
        if "error" in result:
            print(f"WARNING: {result['id']} failed: {result['error']}", file=sys.stderr)
        if out_file is not None:
            out_file.write(json.dumps(result) + "\n")
            out_file.flush()

    try:
        results = evaluate_dataset(rows_to_evaluate, generate_fn, retrieval_corpus=dataset_rows, on_result=on_result)
    finally:
        if out_file is not None:
            out_file.close()

    if len(results) > 1:
        print(json.dumps(aggregate_scores(results), indent=2))
    else:
        print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import statistics
from pathlib import Path

from evaluator.rubric import DIMENSION_KEYS


def pearson_correlation(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    std_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if std_x == 0 or std_y == 0:
        return float("nan")
    return cov / (std_x * std_y)


def load_jsonl(path: str) -> list:
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_agreement(eval_results: list, human_labels: list) -> dict:
    human_by_id = {row["id"]: row["human_scores"] for row in human_labels}
    agreement = {}
    for key in DIMENSION_KEYS:
        judge_scores, human_scores = [], []
        for r in eval_results:
            if r["id"] in human_by_id:
                judge_scores.append(r["judge"]["dimensions"][key]["score"])
                human_scores.append(human_by_id[r["id"]][key])
        if not judge_scores:
            continue
        within_one = sum(1 for j, h in zip(judge_scores, human_scores) if abs(j - h) <= 1) / len(judge_scores)
        agreement[key] = {
            "n": len(judge_scores),
            "pearson_r": round(pearson_correlation(judge_scores, human_scores), 3),
            "pct_within_1": round(within_one * 100, 1),
        }
    return agreement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", default="results/eval_results.jsonl")
    parser.add_argument("--human-labels", default="evaluator/calibration/human_labels.jsonl")
    args = parser.parse_args()

    eval_results = load_jsonl(args.eval_results)
    human_labels = load_jsonl(args.human_labels)
    agreement = compute_agreement(eval_results, human_labels)
    print(json.dumps(agreement, indent=2))


if __name__ == "__main__":
    main()

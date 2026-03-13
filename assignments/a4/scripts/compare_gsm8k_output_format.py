"""
Compare output-format failure modes across GSM8K analysis CSVs.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def classify_output_format(row):
    completion = (row.get("completion") or "").strip()
    is_correct = row.get("is_correct") == "1"
    has_marker = row.get("completion_has_final_marker") == "1"
    has_pred = bool((row.get("pred_answer") or "").strip())
    number_count = int(row.get("completion_number_count") or 0)

    if not completion:
        return "empty_completion"
    if is_correct:
        if has_marker:
            return "correct_with_marker"
        return "correct_without_marker"
    if has_marker and has_pred:
        return "marker_wrong_number"
    if has_marker and not has_pred:
        return "marker_unparseable"
    if number_count > 0:
        return "number_without_marker"
    return "text_without_number"


def grouped_bar(data_by_run, title, output_path):
    labels = list(data_by_run.keys())
    categories = sorted({k for counts in data_by_run.values() for k in counts})
    if not categories:
        return

    width = 0.8 / max(len(labels), 1)
    x_positions = list(range(len(categories)))
    plt.figure(figsize=(12, 6))
    for idx, label in enumerate(labels):
        counts = data_by_run[label]
        total = sum(counts.values()) or 1
        values = [counts.get(cat, 0) / total for cat in categories]
        offsets = [x + (idx - (len(labels) - 1) / 2) * width for x in x_positions]
        plt.bar(offsets, values, width=width, label=label)
    plt.xticks(x_positions, categories, rotation=30, ha="right")
    plt.ylabel("Fraction of examples")
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare GSM8K output-format failure modes across runs")
    parser.add_argument("--csv", action="append", required=True, help="records.csv path")
    parser.add_argument("--label", action="append", required=True, help="label for each csv")
    parser.add_argument("--output-dir", required=True, help="directory for outputs")
    parser.add_argument("--top-examples", type=int, default=10, help="number of example rows to include per bucket")
    args = parser.parse_args()

    if len(args.csv) != len(args.label):
        raise ValueError("--csv and --label must have the same number of values")

    os.makedirs(args.output_dir, exist_ok=True)

    bucket_counts = {}
    bucket_examples = {}
    for label, path in zip(args.label, args.csv):
        rows = load_rows(path)
        counts = Counter()
        examples = {}
        for row in rows:
            bucket = classify_output_format(row)
            counts[bucket] += 1
            examples.setdefault(bucket, [])
            if len(examples[bucket]) < args.top_examples:
                examples[bucket].append({
                    "idx": row.get("idx"),
                    "question": row.get("question", ""),
                    "gold_answer": row.get("gold_answer", ""),
                    "pred_answer": row.get("pred_answer", ""),
                    "completion": row.get("completion", ""),
                    "error_type": row.get("error_type", ""),
                    "answer_style": row.get("answer_style", ""),
                })
        bucket_counts[label] = counts
        bucket_examples[label] = examples

    grouped_bar(
        bucket_counts,
        "GSM8K output-format breakdown",
        os.path.join(args.output_dir, "output_format_breakdown.png"),
    )

    summary_path = os.path.join(args.output_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# GSM8K Output Format Comparison\n\n")
        for label in args.label:
            counts = bucket_counts[label]
            total = sum(counts.values()) or 1
            f.write(f"## {label}\n\n")
            for bucket, count in counts.most_common():
                f.write(f"- {bucket}: {count} ({count / total:.4f})\n")
            f.write("\n")
            for bucket, items in bucket_examples[label].items():
                f.write(f"### {bucket}\n\n")
                for item in items:
                    question = item["question"].replace("\n", " ").strip()
                    completion = item["completion"].replace("\n", " ").strip()
                    f.write(f"- idx={item['idx']} | error={item['error_type']} | style={item['answer_style']} | "
                            f"gold={item['gold_answer']} | pred={item['pred_answer']}\n")
                    f.write(f"  question: {question}\n")
                    f.write(f"  completion: {completion[:240]}\n")
                f.write("\n")

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

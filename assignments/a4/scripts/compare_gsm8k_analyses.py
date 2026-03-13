"""
Compare GSM8K analysis CSVs across multiple checkpoints and summarize error patterns.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


METADATA_COLUMNS = {
    "idx",
    "split",
    "question",
    "gold_answer",
    "pred_answer",
    "is_correct",
    "error_type",
    "answer_style",
    "problem_cluster",
    "question_word_count",
    "question_number_count",
    "completion_word_count",
    "completion_number_count",
    "completion_has_final_marker",
    "completion",
}


def load_records(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["idx"] = int(row["idx"])
            row["is_correct"] = int(row["is_correct"])
            row["question_word_count"] = int(row["question_word_count"])
            row["question_number_count"] = int(row["question_number_count"])
            row["completion_word_count"] = int(row["completion_word_count"])
            row["completion_number_count"] = int(row["completion_number_count"])
            row["completion_has_final_marker"] = int(row["completion_has_final_marker"])
            for key, value in list(row.items()):
                if key in METADATA_COLUMNS:
                    continue
                row[key] = int(value)
            rows.append(row)
        return rows


def grouped_bar(data_by_run, title, output_path, normalize=False):
    labels = list(data_by_run.keys())
    categories = sorted({key for counts in data_by_run.values() for key in counts})
    if not categories:
        return

    width = 0.8 / max(len(labels), 1)
    x_positions = list(range(len(categories)))
    plt.figure(figsize=(12, 6))
    for idx, label in enumerate(labels):
        counts = data_by_run[label]
        values = [counts.get(cat, 0) for cat in categories]
        if normalize:
            total = sum(values) or 1
            values = [v / total for v in values]
        offsets = [x + (idx - (len(labels) - 1) / 2) * width for x in x_positions]
        plt.bar(offsets, values, width=width, label=label)
    plt.xticks(x_positions, categories, rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_path, dpi=180)
    plt.close()


def heatmap(matrix, row_labels, col_labels, title, output_path):
    if not row_labels or not col_labels:
        return
    values = [[matrix[row].get(col, 0.0) for col in col_labels] for row in row_labels]
    plt.figure(figsize=(12, 5))
    im = plt.imshow(values, aspect="auto", cmap="viridis")
    plt.colorbar(im)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare GSM8K analysis outputs across runs")
    parser.add_argument("--csv", action="append", required=True, help="records.csv path")
    parser.add_argument("--label", action="append", required=True, help="label for each csv")
    parser.add_argument("--output-dir", required=True, help="directory for comparison outputs")
    parser.add_argument("--top-disagreements", type=int, default=20, help="number of disagreement examples to include")
    args = parser.parse_args()

    if len(args.csv) != len(args.label):
        raise ValueError("--csv and --label must have the same number of values")

    os.makedirs(args.output_dir, exist_ok=True)

    records_by_label = {}
    indexed = {}
    for label, path in zip(args.label, args.csv):
        records = load_records(path)
        records_by_label[label] = records
        indexed[label] = {row["idx"]: row for row in records}

    accuracy = {}
    error_counts = {}
    answer_style_counts = {}
    cluster_accuracy = defaultdict(dict)
    complexity_accuracy = defaultdict(dict)

    for label, records in records_by_label.items():
        total = len(records) or 1
        accuracy[label] = sum(row["is_correct"] for row in records) / total
        error_counts[label] = Counter(row["error_type"] for row in records)
        answer_style_counts[label] = Counter(row["answer_style"] for row in records)

        cluster_totals = Counter()
        cluster_correct = Counter()
        complexity_totals = Counter()
        complexity_correct = Counter()
        for row in records:
            cluster = row["problem_cluster"]
            cluster_totals[cluster] += 1
            cluster_correct[cluster] += row["is_correct"]

            if row["question_number_count"] <= 2:
                bucket = "simple"
            elif row["question_number_count"] <= 4:
                bucket = "medium"
            else:
                bucket = "complex"
            complexity_totals[bucket] += 1
            complexity_correct[bucket] += row["is_correct"]

        for cluster, total in cluster_totals.items():
            cluster_accuracy[label][cluster] = cluster_correct[cluster] / total
        for bucket, total in complexity_totals.items():
            complexity_accuracy[label][bucket] = complexity_correct[bucket] / total

    grouped_bar({"accuracy": accuracy}, "Accuracy by run", os.path.join(args.output_dir, "accuracy.png"))
    grouped_bar(error_counts, "Error type distribution", os.path.join(args.output_dir, "error_types_by_run.png"), normalize=True)
    grouped_bar(answer_style_counts, "Answer style distribution", os.path.join(args.output_dir, "answer_styles_by_run.png"), normalize=True)

    cluster_cols = sorted({cluster for values in cluster_accuracy.values() for cluster in values})
    heatmap(cluster_accuracy, list(records_by_label.keys()), cluster_cols, "Accuracy by problem cluster", os.path.join(args.output_dir, "problem_cluster_heatmap.png"))
    heatmap(complexity_accuracy, list(records_by_label.keys()), ["simple", "medium", "complex"], "Accuracy by question complexity", os.path.join(args.output_dir, "complexity_heatmap.png"))

    labels = list(records_by_label.keys())
    all_indices = sorted(set().union(*(rows.keys() for rows in indexed.values())))
    disagreements = []
    for idx in all_indices:
        rows = {label: indexed[label].get(idx) for label in labels}
        if any(row is None for row in rows.values()):
            continue
        correct_values = {row["is_correct"] for row in rows.values()}
        if len(correct_values) == 1:
            continue
        score = sum(row["is_correct"] for row in rows.values())
        disagreements.append((abs(score - len(labels) / 2), idx, rows))
    disagreements.sort(reverse=True)

    summary_path = os.path.join(args.output_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# GSM8K Run Comparison\n\n")
        f.write("## Overall Accuracy\n\n")
        for label in labels:
            f.write(f"- {label}: {accuracy[label]:.4f}\n")
        f.write("\n## Problem Cluster Accuracy\n\n")
        for label in labels:
            f.write(f"### {label}\n")
            for cluster, value in sorted(cluster_accuracy[label].items(), key=lambda item: item[1], reverse=True):
                f.write(f"- {cluster}: {value:.4f}\n")
        f.write("\n## Question Complexity Accuracy\n\n")
        for label in labels:
            f.write(f"### {label}\n")
            for bucket in ["simple", "medium", "complex"]:
                value = complexity_accuracy[label].get(bucket, 0.0)
                f.write(f"- {bucket}: {value:.4f}\n")
        f.write("\n## Top Disagreement Examples\n\n")
        for _, idx, rows in disagreements[:args.top_disagreements]:
            sample = rows[labels[0]]
            f.write(f"### Example {idx}\n")
            f.write(f"- Question: {sample['question']}\n")
            f.write(f"- Gold answer: {sample['gold_answer']}\n")
            for label in labels:
                row = rows[label]
                f.write(
                    f"- {label}: correct={row['is_correct']} | error={row['error_type']} | "
                    f"style={row['answer_style']} | pred={row['pred_answer']}\n"
                )
            f.write("\n")

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

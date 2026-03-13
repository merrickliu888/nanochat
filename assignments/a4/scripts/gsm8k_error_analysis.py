"""
Run deterministic GSM8K inference and summarize correct/incorrect patterns.
"""

import argparse
import csv
import os
import re
from decimal import Decimal, InvalidOperation

import matplotlib.pyplot as plt
import torch

from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K, extract_answer


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

CATEGORY_KEYWORDS = {
    "money": ["$", "dollar", "dollars", "cent", "cents", "price", "cost"],
    "time": ["minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years"],
    "distance": ["mile", "miles", "meter", "meters", "kilometer", "kilometers", "km", "foot", "feet", "yard", "yards"],
    "geometry": ["area", "perimeter", "radius", "diameter", "rectangle", "triangle", "circle", "square"],
    "counting": ["how many", "total number", "count", "altogether"],
    "ratio_percent": ["ratio", "percent", "%", "fraction", "half", "double", "twice", "each", "per", "every"],
}


def assistant_parts_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def parse_decimal(text):
    if text is None:
        return None
    try:
        return Decimal(text.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def categorize_question(question):
    lower = question.lower()
    categories = {
        "multi_step": int(len(NUMBER_RE.findall(question)) >= 3),
    }
    for name, keywords in CATEGORY_KEYWORDS.items():
        categories[name] = int(any(keyword in lower for keyword in keywords))
    return categories


def assign_problem_cluster(question, categories):
    if categories.get("geometry"):
        return "geometry"
    if categories.get("money") and categories.get("ratio_percent"):
        return "money_rate"
    if categories.get("time") and categories.get("ratio_percent"):
        return "time_rate"
    if categories.get("counting"):
        return "counting"
    if categories.get("distance"):
        return "distance"
    if categories.get("ratio_percent"):
        return "ratio_percent"
    if categories.get("multi_step"):
        return "multi_step_other"
    return "other"


def classify_error(is_correct, pred_num, gold_num):
    if is_correct:
        return "correct"
    if pred_num is None:
        return "no_numeric_answer"
    if gold_num is None:
        return "bad_reference"

    pred_val = parse_decimal(pred_num)
    gold_val = parse_decimal(gold_num)
    if pred_val is None or gold_val is None:
        return "format_error"
    if pred_val == gold_val:
        return "correct"
    if pred_val == gold_val + 1 or pred_val == gold_val - 1:
        return "off_by_one"
    if pred_val == -gold_val:
        return "sign_error"
    if gold_val != 0:
        rel_err = abs(pred_val - gold_val) / abs(gold_val)
        if rel_err <= Decimal("0.10"):
            return "near_miss"
    larger = max(abs(pred_val), abs(gold_val), Decimal("1"))
    smaller = max(min(abs(pred_val), abs(gold_val)), Decimal("1e-9"))
    if larger / smaller >= 10:
        return "magnitude_error"
    return "wrong_numeric"


def classify_answer_style(completion, pred_num):
    text = completion.strip()
    word_count = len(text.split())
    has_marker = "####" in completion
    has_number = bool(NUMBER_RE.search(completion))
    has_tool_trace = "<<" in completion or ">>" in completion

    if not text:
        return "empty"
    if pred_num is None and has_number:
        return "number_without_marker"
    if pred_num is None:
        return "no_numeric_answer"
    if has_tool_trace and has_marker:
        return "tool_trace_with_final"
    if has_tool_trace:
        return "tool_trace_no_final"
    if has_marker and word_count <= 12:
        return "concise_final"
    if has_marker:
        return "worked_then_final"
    return "other_numeric"


def write_bar_plot(counts, title, output_path):
    if not counts:
        return
    keys = list(counts.keys())
    values = [counts[key] for key in keys]
    plt.figure(figsize=(10, 5))
    plt.bar(keys, values)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def preview(text, limit=120):
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main():
    parser = argparse.ArgumentParser(description="Analyze GSM8K correctness/error patterns for a nanochat checkpoint")
    parser.add_argument("--source", required=True, choices=["sft", "rl"], help="checkpoint source to analyze")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="GSM8K split")
    parser.add_argument("--model-tag", type=str, default=None, help="model tag to load")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step to load")
    parser.add_argument("--max-examples", type=int, default=200, help="max GSM8K examples to analyze")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="generation cap")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory")
    parser.add_argument("--progress-every", type=int, default=10, help="print progress every N examples")
    parser.add_argument("--log-example-every", type=int, default=1, help="print a compact example summary every N completed examples")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    if ddp:
        raise RuntimeError("gsm8k_error_analysis.py is intended for a single process/device run")

    model, tokenizer, _ = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)
    task = GSM8K(subset="main", split=args.split)

    base_dir = get_base_dir()
    model_part = args.model_tag or "latest"
    step_part = f"step_{args.step}" if args.step is not None else "step_latest"
    output_dir = args.output_dir or os.path.join(base_dir, "analysis", f"gsm8k_{args.source}_{model_part}_{args.split}_{step_part}")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "records.csv")
    summary_path = os.path.join(output_dir, "summary.md")

    error_counts = {}
    category_totals = {}
    category_correct = {}
    cluster_totals = {}
    cluster_correct = {}
    answer_style_counts = {}
    header_written = False

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = None
        total_examples = min(args.max_examples, len(task))
        print0(f"Starting GSM8K analysis for {total_examples} examples; writing to {csv_path}")
        for idx in range(total_examples):
            conversation = task[idx]
            prompt = tokenizer.render_for_completion(conversation)
            generated, _ = engine.generate_batch(
                prompt,
                num_samples=1,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            completion = tokenizer.decode(generated[0][len(prompt):])
            is_correct = int(task.evaluate(conversation, completion))
            question = conversation["messages"][0]["content"]
            gold_text = assistant_parts_to_text(conversation["messages"][-1]["content"])
            gold_num = extract_answer(gold_text)
            pred_num = extract_answer(completion)
            categories = categorize_question(question)
            problem_cluster = assign_problem_cluster(question, categories)
            error_type = classify_error(bool(is_correct), pred_num, gold_num)
            answer_style = classify_answer_style(completion, pred_num)

            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            answer_style_counts[answer_style] = answer_style_counts.get(answer_style, 0) + 1
            cluster_totals[problem_cluster] = cluster_totals.get(problem_cluster, 0) + 1
            cluster_correct[problem_cluster] = cluster_correct.get(problem_cluster, 0) + is_correct
            for name, present in categories.items():
                if not present:
                    continue
                category_totals[name] = category_totals.get(name, 0) + 1
                category_correct[name] = category_correct.get(name, 0) + is_correct

            row = {
                "idx": idx,
                "split": args.split,
                "question": question,
                "gold_answer": gold_num or "",
                "pred_answer": pred_num or "",
                "is_correct": is_correct,
                "error_type": error_type,
                "answer_style": answer_style,
                "problem_cluster": problem_cluster,
                "question_word_count": len(question.split()),
                "question_number_count": len(NUMBER_RE.findall(question)),
                "completion_word_count": len(completion.split()),
                "completion_number_count": len(NUMBER_RE.findall(completion)),
                "completion_has_final_marker": int("####" in completion),
                **categories,
                "completion": completion,
            }
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                f.flush()
                os.fsync(f.fileno())
                header_written = True
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            if (idx + 1) % max(args.log_example_every, 1) == 0:
                print0(
                    f"[example {idx + 1}/{total_examples}] correct={is_correct} "
                    f"error={error_type} cluster={problem_cluster} style={answer_style} "
                    f"gold={gold_num or '<none>'} pred={pred_num or '<none>'} "
                    f"question=\"{preview(question)}\" "
                    f"completion=\"{preview(completion)}\""
                )
            if (idx + 1) % max(args.progress_every, 1) == 0 or idx + 1 == total_examples:
                print0(
                    f"Processed {idx + 1}/{total_examples} examples | "
                    f"current accuracy={(error_counts.get('correct', 0) / (idx + 1)):.4f}"
                )

        if not header_written:
            # If no rows were written, still emit an empty CSV with headers for easier debugging.
            empty_row = {
                "idx": 0,
                "split": args.split,
                "question": "",
                "gold_answer": "",
                "pred_answer": "",
                "is_correct": 0,
                "error_type": "",
                "answer_style": "",
                "problem_cluster": "",
                "question_word_count": 0,
                "question_number_count": 0,
                "completion_word_count": 0,
                "completion_number_count": 0,
                "completion_has_final_marker": 0,
                **{name: 0 for name in categorize_question("")},
                "completion": "",
            }
            writer = csv.DictWriter(f, fieldnames=list(empty_row.keys()))
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

    total = sum(error_counts.values())
    accuracy = error_counts.get("correct", 0) / total if total else 0.0
    category_accuracy = {
        name: category_correct.get(name, 0) / total_count
        for name, total_count in category_totals.items()
        if total_count > 0
    }
    cluster_accuracy = {
        name: cluster_correct.get(name, 0) / total_count
        for name, total_count in cluster_totals.items()
        if total_count > 0
    }

    write_bar_plot(error_counts, "GSM8K error types", os.path.join(output_dir, "error_types.png"))
    write_bar_plot(category_accuracy, "GSM8K accuracy by category", os.path.join(output_dir, "category_accuracy.png"))
    write_bar_plot(cluster_accuracy, "GSM8K accuracy by problem cluster", os.path.join(output_dir, "problem_cluster_accuracy.png"))
    write_bar_plot(answer_style_counts, "GSM8K answer styles", os.path.join(output_dir, "answer_styles.png"))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# GSM8K Error Analysis ({args.source}, {args.split})\n\n")
        f.write(f"- Examples: {total}\n")
        f.write(f"- Accuracy: {accuracy:.4f}\n\n")
        f.write("## Error Types\n\n")
        for name, count in sorted(error_counts.items(), key=lambda item: item[1], reverse=True):
            f.write(f"- {name}: {count}\n")
        f.write("\n## Category Accuracy\n\n")
        for name, value in sorted(category_accuracy.items(), key=lambda item: item[1], reverse=True):
            f.write(f"- {name}: {value:.4f} ({category_correct.get(name, 0)}/{category_totals.get(name, 0)})\n")
        f.write("\n## Problem Cluster Accuracy\n\n")
        for name, value in sorted(cluster_accuracy.items(), key=lambda item: item[1], reverse=True):
            f.write(f"- {name}: {value:.4f} ({cluster_correct.get(name, 0)}/{cluster_totals.get(name, 0)})\n")
        f.write("\n## Answer Styles\n\n")
        for name, count in sorted(answer_style_counts.items(), key=lambda item: item[1], reverse=True):
            f.write(f"- {name}: {count}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    compute_cleanup()


if __name__ == "__main__":
    with torch.no_grad():
        main()

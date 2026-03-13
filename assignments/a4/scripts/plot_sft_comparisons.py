"""
Generate clearer comparison plots for the SFT report.

- Section 1: grouped bar chart for pretrained/base-chat-eval vs baseline SFT
- Section 2: grouped bar chart for baseline SFT vs added-dataset SFT runs

Inputs are the W&B summary JSON exports already stored under report_artifacts/wandb.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


TASK_METRICS = [
    "chatcore_metric",
    "chatcore/ARC-Easy",
    "chatcore/ARC-Challenge",
    "chatcore/MMLU",
    "chatcore/GSM8K",
    "chatcore/HumanEval",
    "chatcore/SpellingBee",
]

METRIC_LABELS = {
    "chatcore_metric": "ChatCORE",
    "chatcore/ARC-Easy": "ARC-Easy",
    "chatcore/ARC-Challenge": "ARC-Challenge",
    "chatcore/MMLU": "MMLU",
    "chatcore/GSM8K": "GSM8K",
    "chatcore/HumanEval": "HumanEval",
    "chatcore/SpellingBee": "SpellingBee",
}


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_values(summary: dict) -> list[float]:
    source = summary["summary"]
    values: list[float] = []
    for metric in TASK_METRICS:
        value = source.get(metric)
        if value is None:
            raise KeyError(f"Metric {metric} missing from {summary.get('path', '<unknown>')}")
        values.append(float(value))
    return values


def _plot_grouped_bars(
    summaries: list[dict],
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    width = 0.8 / max(len(summaries), 1)
    x_positions = list(range(len(TASK_METRICS)))

    plt.figure(figsize=(12, 6))
    for idx, summary in enumerate(summaries):
        offsets = [x + (idx - (len(summaries) - 1) / 2) * width for x in x_positions]
        plt.bar(offsets, _metric_values(summary), width=width, label=labels[idx])

    plt.xticks(x_positions, [METRIC_LABELS[m] for m in TASK_METRICS], rotation=25, ha="right")
    plt.ylabel("Accuracy / centered score")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Wrote {output_path}")

# def _plot_grouped_bars(
#     summaries: list[dict],
#     labels: list[str],
#     title: str,
#     output_path: Path,
# ) -> None:
#     width = 0.8 / max(len(summaries), 1)
#     x_positions = list(range(len(TASK_METRICS)))

#     all_values: list[float] = []
#     series_values: list[list[float]] = []
#     for summary in summaries:
#         values = _metric_values(summary)
#         series_values.append(values)
#         all_values.extend(values)

#     y_min = min(all_values)
#     y_max = max(all_values)

#     lower = min(-0.05, y_min - 0.02) if y_min < 0 else 0.0
#     upper = max(1.05, y_max + 0.02)

#     plt.figure(figsize=(12, 6))
#     for idx, values in enumerate(series_values):
#         offsets = [x + (idx - (len(series_values) - 1) / 2) * width for x in x_positions]
#         plt.bar(offsets, values, width=width, label=labels[idx])

#     plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
#     plt.xticks(x_positions, [METRIC_LABELS[m] for m in TASK_METRICS], rotation=25, ha="right")
#     plt.ylabel("Accuracy or normalized score")
#     plt.title(title)
#     plt.ylim(lower, upper)
#     plt.legend()
#     plt.tight_layout()
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(output_path, dpi=180)
#     plt.close()
#     print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bar-chart comparison plots for SFT reporting")
    parser.add_argument("--base-chat", type=Path, default=None, help="summary JSON for base-chat evaluation run")
    parser.add_argument("--baseline-sft", type=Path, required=True, help="summary JSON for baseline SFT run")
    parser.add_argument("--ultrachat-sft", type=Path, default=None, help="summary JSON for ultrachat SFT run")
    parser.add_argument("--tulu-if-sft", type=Path, default=None, help="summary JSON for tulu-if SFT run")
    parser.add_argument("--tulu-math-sft", type=Path, default=None, help="summary JSON for tulu-math SFT run")
    parser.add_argument("--output-dir", type=Path, default=Path("report_artifacts/plots/final_compare"))
    args = parser.parse_args()

    baseline = _load_summary(args.baseline_sft)

    # if args.base_chat is not None:
    #     base_chat = _load_summary(args.base_chat)
    #     _plot_grouped_bars(
    #         [base_chat, baseline],
    #         ["pretrained", "sft baseline"],
    #         "Pretrained vs Baseline SFT",
    #         args.output_dir / "section1_base_vs_baseline.png",
    #     )

    ablation_summaries = [baseline]
    ablation_labels = ["sft baseline"]
    if args.ultrachat_sft is not None:
        ablation_summaries.append(_load_summary(args.ultrachat_sft))
        ablation_labels.append("sft + ultrachat")
    if args.tulu_if_sft is not None:
        ablation_summaries.append(_load_summary(args.tulu_if_sft))
        ablation_labels.append("sft + tulu-if")
    if args.tulu_math_sft is not None:
        ablation_summaries.append(_load_summary(args.tulu_math_sft))
        ablation_labels.append("sft + tulu-math")
    if len(ablation_summaries) > 1:
        _plot_grouped_bars(
            ablation_summaries,
            ablation_labels,
            "Baseline SFT vs Added-Dataset SFT",
            args.output_dir / "section2_baseline_vs_augmented.png",
        )


if __name__ == "__main__":
    main()

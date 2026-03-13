"""
Download run histories from Weights & Biases and plot selected metrics.
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import wandb

from nanochat.common import get_base_dir


def rolling_mean(values, window):
    if window <= 1:
        return values
    out = []
    acc = 0.0
    buf = []
    for value in values:
        buf.append(value)
        acc += value
        if len(buf) > window:
            acc -= buf.pop(0)
        out.append(acc / len(buf))
    return out


def fetch_metric_series(run, metric, x_key):
    xs = []
    ys = []
    for row in run.scan_history(keys=[x_key, metric]):
        x = row.get(x_key)
        y = row.get(metric)
        if x is None or y is None:
            continue
        if isinstance(y, float) and (math.isnan(y) or math.isinf(y)):
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def main():
    parser = argparse.ArgumentParser(description="Compare selected W&B metrics across runs")
    parser.add_argument("--run", action="append", required=True, help="W&B run path entity/project/run_id_or_name; can be passed multiple times")
    parser.add_argument("--label", action="append", default=[], help="plot label for each run, in the same order as --run")
    parser.add_argument("--metric", action="append", required=True, help="metric to plot; can be passed multiple times")
    parser.add_argument("--x-key", type=str, default="step", help="history key for x-axis")
    parser.add_argument("--smooth-window", type=int, default=1, help="simple rolling mean window")
    parser.add_argument("--output-dir", type=str, default=None, help="directory for PNG plots")
    args = parser.parse_args()

    if args.label and len(args.label) != len(args.run):
        raise ValueError("If --label is provided, it must be passed once per --run")

    api = wandb.Api()
    labels = args.label or args.run

    output_dir = args.output_dir or os.path.join(get_base_dir(), "analysis", "wandb_compare")
    os.makedirs(output_dir, exist_ok=True)

    runs = []
    for run_path, label in zip(args.run, labels):
        runs.append((api.run(run_path), label))

    for metric in args.metric:
        plt.figure(figsize=(10, 6))
        plotted = False
        for run, label in runs:
            xs, ys = fetch_metric_series(run, metric, args.x_key)
            if not xs:
                continue
            ys = rolling_mean(ys, args.smooth_window)
            plt.plot(xs, ys, label=label)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.xlabel(args.x_key)
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{metric.replace('/', '_')}.png")
        plt.savefig(output_path, dpi=180)
        plt.close()
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

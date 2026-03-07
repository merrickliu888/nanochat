"""
Extract per-step training loss from a pretrain run log and plot a learning curve.

Usage:
  python -m scripts.plot_pretrain_loss /path/to/pretrain.log \
      --output-prefix /tmp/pretrain_loss
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


STEP_LOSS_RE = re.compile(
    r"^\s*step\s+(?P<step>\d+)\s*/\s*\d+\s*\(.*?\)\s*\|\s*loss:\s*(?P<loss>-?\d+\.\d+(?:[eE][+-]?\d+)?)"
)
VAL_BPB_RE = re.compile(r"^\s*Validation bpb:\s*(?P<val_bpb>\d+\.\d+(?:[eE][+-]?\d+)?|nan|inf|NaN|INF)")


def _extract_from_file(path: Path) -> tuple[list[int], list[float], list[tuple[int, float]]]:
    steps: list[int] = []
    losses: list[float] = []
    val_bpb_records: list[tuple[int, float]] = []
    current_step = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            step_match = STEP_LOSS_RE.match(line)
            if step_match:
                current_step = int(step_match.group("step"))
                losses.append(float(step_match.group("loss")))
                steps.append(current_step)
                continue

            val_match = VAL_BPB_RE.match(line)
            if val_match and current_step is not None:
                val_text = val_match.group("val_bpb").lower()
                if val_text in {"nan", "inf", "nan", "infty"}:
                    continue
                val_bpb_records.append((current_step, float(val_text)))

    return steps, losses, val_bpb_records


def _iter_log_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for p in sorted(path.rglob("*.log")):
            yield p
        for p in sorted(path.rglob("*.txt")):
            yield p
        return
    raise FileNotFoundError(f"Input path does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-step train/loss from a pretrain run.")
    parser.add_argument("log_path", type=Path, help="Path to a pretrain log file or directory containing logs.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("pretrain_loss"),
        help="Output stem for CSV and PNG (default: pretrain_loss)",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    parser.add_argument("--min-step", type=int, default=None, help="Ignore steps before this value.")
    parser.add_argument("--max-step", type=int, default=None, help="Ignore steps after this value.")

    args = parser.parse_args()
    all_steps: list[int] = []
    all_losses: list[float] = []
    all_val_bpb: list[tuple[int, float]] = []

    for log_file in _iter_log_files(args.log_path):
        steps, losses, val_bpb = _extract_from_file(log_file)
        all_steps.extend(steps)
        all_losses.extend(losses)
        all_val_bpb.extend(val_bpb)

    if not all_steps:
        raise RuntimeError("No step-loss lines found. Expected lines like: step 00010/xxxxx (...) | loss: 0.123456")

    # Keep as observed order, then stable-sort by step to avoid duplicate overlaps.
    pairs = sorted(
        ((s, l) for s, l in zip(all_steps, all_losses)),
        key=lambda x: x[0],
    )
    steps = [s for s, _ in pairs]
    losses = [l for _, l in pairs]

    if args.min_step is not None:
        filtered = [(s, l) for s, l in zip(steps, losses) if s >= args.min_step]
        steps, losses = zip(*filtered) if filtered else ([], [])
    if args.max_step is not None:
        filtered = [(s, l) for s, l in zip(steps, losses) if s <= args.max_step]
        steps, losses = zip(*filtered) if filtered else ([], [])

    if not steps:
        raise RuntimeError("Filtering removed all data points.")

    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss"])
        for s, l in zip(steps, losses):
            writer.writerow([s, f"{l:.12f}"])
    print(f"Wrote {len(steps)} points to {csv_path}")

    import matplotlib.pyplot as plt  # imported lazily

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label="train/loss", linewidth=1.2)
    if all_val_bpb:
        val_steps, val_losses = zip(*sorted(all_val_bpb))
        if args.min_step is not None:
            filtered_val = [(s, v) for s, v in zip(val_steps, val_losses) if s >= args.min_step]
            val_steps, val_losses = zip(*filtered_val) if filtered_val else ([], [])
        if args.max_step is not None:
            filtered_val = [(s, v) for s, v in zip(val_steps, val_losses) if s <= args.max_step]
            val_steps, val_losses = zip(*filtered_val) if filtered_val else ([], [])
        if val_steps:
            plt.twinx()
            plt.plot(val_steps, val_losses, label="val bpb", linestyle="--", linewidth=1.0, color="orange")

    plt.xlabel("Step")
    plt.ylabel("Debiased train loss")
    plt.title("Pretrain loss curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = args.output_prefix.with_suffix(".png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

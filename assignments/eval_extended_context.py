"""
Evaluate picochat baseline vs YaRN at extended context lengths.

Two experiment modes (set EXPERIMENT env var):

  EXPERIMENT=extended  (default)
    Loads pico-baseline and pico-yarn (both trained at seq_len=512) and
    evaluates validation BPB at seq_len = 512, 1024, 2048.

  EXPERIMENT=seqlen
    Loads seqlen{N}-baseline and seqlen{N}-yarn checkpoints (each trained AT
    seq_len=N) and evaluates each model at its own training seq_len.
    This is the correct YaRN vs baseline comparison: full causal attention at
    the target width, so RoPE positions are genuinely out-of-distribution for
    the baseline but compressed-in-distribution for YaRN.

Usage (run from repo root after downloading checkpoints from Modal volume):
    python assignments/eval_extended_context.py
    EXPERIMENT=seqlen python assignments/eval_extended_context.py

Optional env vars:
    EXPERIMENT          "extended" or "seqlen"            (default: extended)
    PICO_BASELINE_TAG   model-tag for extended baseline   (default: pico-baseline)
    PICO_YARN_TAG       model-tag for extended yarn       (default: pico-yarn)
    EVAL_SEQ_LENS       seq lens for extended mode        (default: 512,1024,2048)
    SEQLEN_LIST         seq lens for seqlen mode          (default: 512,1024,2048,4096)
    EVAL_TOKENS         tokens used per eval              (default: 1048576)
"""

import os
import math
import torch

from nanochat.common import get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

# ── CONFIG ────────────────────────────────────────────────────────────────────
EXPERIMENT   = os.environ.get("EXPERIMENT", "extended")
BASELINE_TAG = os.environ.get("PICO_BASELINE_TAG", "pico-baseline")
YARN_TAG     = os.environ.get("PICO_YARN_TAG",     "pico-yarn")
SEQ_LENS     = [int(x) for x in os.environ.get("EVAL_SEQ_LENS", "512,1024,2048").split(",")]
SEQLEN_LIST  = [int(x) for x in os.environ.get("SEQLEN_LIST", "512,1024,2048,4096").split(",")]
EVAL_TOKENS  = int(os.environ.get("EVAL_TOKENS", str(1024 * 1024)))
DEVICE_BATCH = 4   # small batch — we are changing seq_len so need to stay in memory

# ── HELPERS ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_bpb_at_seq_len(model, tokenizer, token_bytes, seq_len: int, eval_tokens: int, device) -> float:
    """Evaluate BPB using a val dataloader at the given seq_len."""
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, DEVICE_BATCH, seq_len, split="val", device=device
    )
    steps = max(1, eval_tokens // (DEVICE_BATCH * seq_len))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y = next(iter(loader))
        loss2d = model(x, y, loss_reduction='none')
        loss2d = loss2d.view(-1)
        y_flat = y.view(-1)
        valid = y_flat >= 0
        y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
        nb = torch.where(valid, token_bytes[y_safe], torch.zeros_like(y_flat, dtype=token_bytes.dtype))
        total_nats  += (loss2d * (nb > 0)).sum().item()
        total_bytes += nb.sum().item()
    if total_bytes == 0:
        return float('inf')
    return total_nats / (math.log(2) * total_bytes)


def _load_checkpoint(model_tag: str, seq_len: int, device):
    """Load a checkpoint and extend its rotary cache to cover seq_len."""
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
    model, tokenizer, _ = load_model_from_dir(checkpoints_dir, device, "eval", model_tag=model_tag)
    model.eval()
    head_dim = model.config.n_embd // model.config.n_head
    cos, sin = model._precompute_rotary_embeddings(seq_len * 10, head_dim, device=device)
    model.cos, model.sin = cos, sin
    return model, tokenizer


def load_model_for_eval(model_tag: str, device):
    """Load a model and extend rotary cache to cover the longest SEQ_LENS entry."""
    return _load_checkpoint(model_tag, max(SEQ_LENS), device)


# ── EXPERIMENT: EXTENDED ──────────────────────────────────────────────────────

def run_extended():
    """
    Eval pico-baseline and pico-yarn (both trained at seq_len=512) at
    seq_len = 512, 1024, 2048.  NOTE: this experiment is confounded by
    sliding-window attention — see run_seqlen() for the correct comparison.
    """
    device_type = autodetect_device_type()
    device = torch.device(device_type)

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    print(f"Loading models — device: {device_type}, dtype: {COMPUTE_DTYPE}")
    print(f"Eval seq lens: {SEQ_LENS}  |  Eval tokens each: {EVAL_TOKENS:,}\n")

    models = {}
    for tag in [BASELINE_TAG, YARN_TAG]:
        print(f"  Loading {tag} ...")
        try:
            m, _ = load_model_for_eval(tag, device)
            models[tag] = m
        except FileNotFoundError as e:
            print(f"  WARNING: could not load {tag}: {e}")

    if not models:
        print("No models found. Download checkpoints from the Modal volume first.")
        return

    col_w = 14
    header = f"{'seq_len':>{col_w}}" + "".join(f"{tag:>{col_w}}" for tag in models)
    print(header)
    print("-" * len(header))

    results = {tag: {} for tag in models}
    for seq_len in SEQ_LENS:
        row = f"{seq_len:>{col_w}}"
        for tag, model in models.items():
            if seq_len > model.cos.size(1):
                head_dim = model.config.n_embd // model.config.n_head
                cos, sin = model._precompute_rotary_embeddings(seq_len * 10, head_dim, device=device)
                model.cos, model.sin = cos, sin
            bpb = eval_bpb_at_seq_len(model, tokenizer, token_bytes, seq_len, EVAL_TOKENS, device)
            results[tag][seq_len] = bpb
            row += f"{bpb:>{col_w}.4f}"
        print(row)

    print()
    print("BPB delta vs baseline at each seq_len (positive = baseline is worse):")
    if BASELINE_TAG in results and YARN_TAG in results:
        for seq_len in SEQ_LENS:
            base_bpb = results[BASELINE_TAG].get(seq_len, float('nan'))
            yarn_bpb = results[YARN_TAG].get(seq_len, float('nan'))
            delta = base_bpb - yarn_bpb
            print(f"  seq_len={seq_len:>5}: baseline={base_bpb:.4f}  yarn={yarn_bpb:.4f}  delta={delta:+.4f}")


# ── EXPERIMENT: SEQLEN ────────────────────────────────────────────────────────

def run_seqlen():
    """
    Load seqlen{N}-baseline and seqlen{N}-yarn checkpoints (each trained AT
    seq_len=N) and evaluate each at its own training seq_len.

    This is the correct comparison: full causal attention at the target width
    means RoPE positions are genuinely used, so YaRN's frequency compression
    has a real effect.
    """
    device_type = autodetect_device_type()
    device = torch.device(device_type)

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    print(f"YaRN vs Baseline: val BPB at training seq_len")
    print(f"Device: {device_type}  |  Eval tokens each: {EVAL_TOKENS:,}\n")

    col_w = 14
    header = f"{'seq_len':>{col_w}}{'baseline':>{col_w}}{'yarn':>{col_w}}{'delta':>{col_w}}"
    print(header)
    print("-" * len(header))

    all_results = {}
    for seq_len in SEQLEN_LIST:
        bpbs = {}
        for variant in ["baseline", "yarn"]:
            tag = f"seqlen{seq_len}-{variant}"
            try:
                print(f"  Loading {tag} ...", flush=True)
                model, _ = _load_checkpoint(tag, seq_len, device)
                bpb = eval_bpb_at_seq_len(model, tokenizer, token_bytes, seq_len, EVAL_TOKENS, device)
                bpbs[variant] = bpb
                del model
                if device_type == "cuda":
                    torch.cuda.empty_cache()
            except FileNotFoundError:
                print(f"  WARNING: checkpoint {tag} not found")
                bpbs[variant] = float('nan')

        base_bpb = bpbs.get("baseline", float('nan'))
        yarn_bpb = bpbs.get("yarn",     float('nan'))
        delta     = base_bpb - yarn_bpb

        base_str  = f"{base_bpb:>{col_w}.4f}" if not math.isnan(base_bpb) else f"{'N/A':>{col_w}}"
        yarn_str  = f"{yarn_bpb:>{col_w}.4f}" if not math.isnan(yarn_bpb) else f"{'N/A':>{col_w}}"
        delta_str = f"{delta:>{col_w}+.4f}"   if not math.isnan(delta)    else f"{'N/A':>{col_w}}"
        print(f"{seq_len:>{col_w}}{base_str}{yarn_str}{delta_str}")
        all_results[seq_len] = bpbs

    print()
    print("delta > 0 means baseline is worse (YaRN helps); delta < 0 means YaRN hurts.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if EXPERIMENT == "seqlen":
        run_seqlen()
    else:
        run_extended()


if __name__ == "__main__":
    main()

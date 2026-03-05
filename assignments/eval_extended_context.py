"""
Evaluate picochat baseline vs YaRN at extended context lengths.

Loads pico-baseline and pico-yarn checkpoints from base_checkpoints/ and
evaluates validation BPB at seq_len = 512, 1024, 2048.

Note: picochat uses window-pattern=L, which sets the attention window to
config.sequence_len (512).  At inference, position IDs therefore never
exceed 512 regardless of the eval batch length, so this eval measures
the effect of *more local context* (each token attends to up to 512 prior
tokens), not true RoPE extrapolation.

Usage (run from repo root after downloading checkpoints from Modal volume):
    python assignments/eval_extended_context.py

Optional env vars:
    PICO_BASELINE_TAG   checkpoint model-tag for baseline   (default: pico-baseline)
    PICO_YARN_TAG       checkpoint model-tag for yarn model (default: pico-yarn)
    EVAL_SEQ_LENS       comma-separated list of seq lens    (default: 512,1024,2048)
    EVAL_TOKENS         tokens used for each eval           (default: 1048576)
"""

import os
import math
import torch

from nanochat.common import get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASELINE_TAG = os.environ.get("PICO_BASELINE_TAG", "pico-baseline")
YARN_TAG     = os.environ.get("PICO_YARN_TAG",     "pico-yarn")
SEQ_LENS     = [int(x) for x in os.environ.get("EVAL_SEQ_LENS", "512,1024,2048").split(",")]
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


def load_model_for_eval(model_tag: str, device):
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
    model, tokenizer, meta = load_model_from_dir(checkpoints_dir, device, "eval", model_tag=model_tag)
    model.eval()
    # Extend rotary cache to cover the longest requested seq_len
    max_seq = max(SEQ_LENS)
    head_dim = model.config.n_embd // model.config.n_head
    rotary_len = max_seq * 10
    cos, sin = model._precompute_rotary_embeddings(rotary_len, head_dim, device=device)
    model.cos, model.sin = cos, sin
    return model, tokenizer


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
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

    # Header
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
                rotary_len = seq_len * 10
                cos, sin = model._precompute_rotary_embeddings(rotary_len, head_dim, device=device)
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


if __name__ == "__main__":
    main()

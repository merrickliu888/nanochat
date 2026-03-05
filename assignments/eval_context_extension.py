"""
Part 3: Evaluate context extension — pico-baseline (512) vs ctx-ext-2048.

Two evaluation tasks:
  1. BPB at seq_len = 512, 1024, 2048
  2. Long-range passkey retrieval (needle-in-a-haystack)

Usage (run from repo root after downloading checkpoints from Modal volume):
    python -m assignments.eval_context_extension

Optional env vars:
    PHASE1_TAG      model-tag for phase 1 checkpoint  (default: pico-baseline)
    PHASE2_TAG      model-tag for phase 2 checkpoint  (default: ctx-ext-2048)
    EVAL_SEQ_LENS   comma-separated seq lens           (default: 512,1024,2048)
    EVAL_TOKENS     tokens used per BPB eval           (default: 1048576)
"""

import os
import math
import random
import torch

from nanochat.common import get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model_from_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.engine import Engine

# ── CONFIG ────────────────────────────────────────────────────────────────────
PHASE1_TAG  = os.environ.get("PHASE1_TAG", "pico-baseline")
PHASE2_TAG  = os.environ.get("PHASE2_TAG", "ctx-ext-2048")
EVAL_SEQ_LENS = [int(x) for x in os.environ.get("EVAL_SEQ_LENS", "512,1024,2048").split(",")]
EVAL_TOKENS = int(os.environ.get("EVAL_TOKENS", str(1024 * 1024)))
DEVICE_BATCH = 4

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _load_checkpoint(model_tag: str, eval_seq_len: int, device):
    """Load a checkpoint and extend its rotary cache to cover eval_seq_len."""
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
    model, tokenizer, _ = load_model_from_dir(
        checkpoints_dir, device, "eval", model_tag=model_tag
    )
    model.eval()
    head_dim = model.config.n_embd // model.config.n_head
    cos, sin = model._precompute_rotary_embeddings(
        eval_seq_len * 10, head_dim, device=device
    )
    model.cos, model.sin = cos, sin
    return model, tokenizer


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
        nb = torch.where(valid, token_bytes[y_safe],
                         torch.zeros_like(y_flat, dtype=token_bytes.dtype))
        total_nats += (loss2d * (nb > 0)).sum().item()
        total_bytes += nb.sum().item()
    if total_bytes == 0:
        return float('inf')
    return total_nats / (math.log(2) * total_bytes)


# ── TASK A: BPB COMPARISON ───────────────────────────────────────────────────

def run_bpb_comparison():
    """Evaluate BPB at multiple seq_lens for both checkpoints."""
    device_type = autodetect_device_type()
    device = torch.device(device_type)
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    print("=" * 60)
    print("Task A: BPB Comparison")
    print("=" * 60)
    print(f"Eval tokens per measurement: {EVAL_TOKENS:,}\n")

    results = {}
    for tag in [PHASE1_TAG, PHASE2_TAG]:
        results[tag] = {}
        max_sl = max(EVAL_SEQ_LENS)
        print(f"Loading {tag} ...", flush=True)
        model, _ = _load_checkpoint(tag, max_sl, device)
        for seq_len in EVAL_SEQ_LENS:
            bpb = eval_bpb_at_seq_len(
                model, tokenizer, token_bytes, seq_len, EVAL_TOKENS, device
            )
            results[tag][seq_len] = bpb
            print(f"  seq_len={seq_len}: BPB={bpb:.4f}")
        del model
        if device_type == "cuda":
            torch.cuda.empty_cache()

    # Print table
    col_w = 16
    print()
    header = f"{'seq_len':>{col_w}}{PHASE1_TAG:>{col_w}}{PHASE2_TAG:>{col_w}}{'delta':>{col_w}}"
    print(header)
    print("-" * len(header))
    for seq_len in EVAL_SEQ_LENS:
        p1 = results[PHASE1_TAG][seq_len]
        p2 = results[PHASE2_TAG][seq_len]
        delta = p1 - p2
        print(f"{seq_len:>{col_w}}{p1:>{col_w}.4f}{p2:>{col_w}.4f}{delta:>+{col_w}.4f}")
    print()
    print("delta > 0 means phase 1 is worse (context extension helps)")


# ── TASK B: PASSKEY RETRIEVAL ────────────────────────────────────────────────

def run_passkey_retrieval():
    """Needle-in-a-haystack passkey retrieval at various depths."""
    device_type = autodetect_device_type()
    device = torch.device(device_type)

    print()
    print("=" * 60)
    print("Task B: Passkey Retrieval")
    print("=" * 60)

    FILLER = "The quick brown fox jumps over the lazy dog. "
    NUM_TRIALS = 20
    POSITIONS = [0.1, 0.5, 0.9]
    random.seed(42)

    for tag in [PHASE1_TAG, PHASE2_TAG]:
        # Use the model's training seq_len as the max context for generation
        max_seq = 2048 if tag == PHASE2_TAG else 512
        print(f"\nModel: {tag} (max_seq={max_seq})")

        model, tokenizer = _load_checkpoint(tag, max_seq, device)
        engine = Engine(model, tokenizer)

        for pos_frac in POSITIONS:
            correct = 0
            for trial in range(NUM_TRIALS):
                passkey = random.randint(10000, 99999)
                needle = f"The secret passkey is {passkey}. Remember this number."

                # Build context with needle at specified position
                # Aim for ~max_seq tokens total (minus room for generation)
                target_ctx_tokens = max_seq - 32
                tokens_before = int(target_ctx_tokens * pos_frac)
                tokens_after = target_ctx_tokens - tokens_before

                filler_before = (FILLER * (tokens_before // 8 + 1))
                filler_after = (FILLER * (tokens_after // 8 + 1))
                prompt = filler_before + needle + filler_after
                prompt += "\nWhat is the secret passkey mentioned above? The passkey is "

                tokens = tokenizer(prompt, prepend="<|bos|>")
                tokens = tokens[:max_seq - 16]  # leave room for generation

                samples, _ = engine.generate_batch(
                    tokens, num_samples=1, max_tokens=16, temperature=0, seed=trial
                )
                output = tokenizer.decode(samples[0][len(tokens):])
                if str(passkey) in output:
                    correct += 1

            acc = correct / NUM_TRIALS * 100
            print(f"  Position {pos_frac:.0%}: {correct}/{NUM_TRIALS} ({acc:.0f}%)")

        del model, engine
        if device_type == "cuda":
            torch.cuda.empty_cache()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    run_bpb_comparison()
    run_passkey_retrieval()


if __name__ == "__main__":
    main()

"""
Modal entrypoints for GSM8K reward-system RL ablations.
"""

import os

import modal

from scripts.modal_train import (
    FINETUNE_TIMEOUT_SEC,
    VOLUME_MOUNT,
    _python,
    _setup_cache,
    _torchrun,
    image,
    secret,
    volume,
)


app = modal.App("nanochat-reward-ablations")

FINETUNE_GPU = "H100:8"
FINETUNE_NPROC = 8
RL_TIMEOUT_SEC = 60 * 60 * 6


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=FINETUNE_GPU,
    timeout=RL_TIMEOUT_SEC,
)
def stage_rl_reward_ablation(
    reward_system: str,
    wandb_run: str,
    source: str = "sft",
    model_tag: str = "",
    model_step: int = -1,
    output_tag: str = "",
    wandb_id: str = "",
    wandb_resume: str = "never",
    embedding_lr: float = -1.0,
    matrix_lr: float = -1.0,
    unembedding_lr: float = -1.0,
) -> None:
    _setup_cache()
    args = [
        f"--reward-system={reward_system}",
        f"--run={wandb_run}",
        f"--source={source}",
        f"--wandb-resume={wandb_resume}",
    ]
    if model_tag:
        args.append(f"--model-tag={model_tag}")
    if model_step >= 0:
        args.append(f"--model-step={model_step}")
    if output_tag:
        args.append(f"--output-tag={output_tag}")
    if wandb_id:
        args.append(f"--wandb-id={wandb_id}")
    if embedding_lr >= 0:
        args.append(f"--embedding-lr={embedding_lr}")
    if matrix_lr >= 0:
        args.append(f"--matrix-lr={matrix_lr}")
    if unembedding_lr >= 0:
        args.append(f"--unembedding-lr={unembedding_lr}")
    _torchrun("assignments.a4.chat_rl_reward_ablation", args, nproc=FINETUNE_NPROC)

    eval_args = ["-i", "rl", "-a", "GSM8K"]
    if output_tag:
        eval_args.extend(["-g", output_tag])
    elif model_tag:
        eval_args.extend(["-g", model_tag])
    _torchrun("scripts.chat_eval", eval_args, nproc=FINETUNE_NPROC)
    volume.commit()


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G:1",
    timeout=60 * 60,
)
def stage_reward_analysis(
    source: str,
    split: str = "test",
    model_tag: str = "",
    step: int = -1,
    max_examples: int = 100,
    output_dir: str = "",
    max_new_tokens: int = 128,
    progress_every: int = 10,
    log_example_every: int = 1,
) -> None:
    _setup_cache()
    args = [
        f"--source={source}",
        f"--split={split}",
        f"--max-examples={max_examples}",
        f"--max-new-tokens={max_new_tokens}",
        f"--progress-every={progress_every}",
        f"--log-example-every={log_example_every}",
    ]
    if model_tag:
        args.append(f"--model-tag={model_tag}")
    if step >= 0:
        args.append(f"--step={step}")
    if output_dir:
        args.append(f"--output-dir={output_dir}")
    _python("assignments.a4.gsm8k_error_analysis", args)
    volume.commit()

"""
Modal workflow for the nanochat pretrain/SFT/RL experiment sequence.

Run with:
    modal run -m assignments.a4.modal_experiments::stage_prepare_pretrain_data
    modal run -m assignments.a4.modal_experiments::stage_train_tokenizer
    modal run -m assignments.a4.modal_experiments::stage_pretrain_original --wandb-run my-pretrain --model-tag d24_modal
    modal run -m assignments.a4.modal_experiments::stage_post_pretrain_eval --model-tag d24_modal
    modal run -m assignments.a4.modal_experiments::stage_sft_original --wandb-run my-sft --model-tag d24_modal --output-tag d24_sft_baseline
    modal run -m assignments.a4.modal_experiments::stage_prepare_extra_sft_dataset --dataset tulu_math
    modal run -m assignments.a4.modal_experiments::stage_sft_extra --dataset tulu_math --wandb-run my-sft-math --model-tag d24_modal --output-tag d24_sft_tulu_math
    modal run -m assignments.a4.modal_experiments::stage_rl_gsm8k --wandb-run my-rl --model-tag d24_sft_baseline --output-tag d24_rl_baseline
"""

import os

import modal

from scripts.modal_train import (
    BASE_DIR,
    DOWNLOAD_TIMEOUT_SEC,
    FINETUNE_TIMEOUT_SEC,
    IDENTITY_JSONL_URL,
    NANOCHAT_CACHE,
    PRETRAIN_TIMEOUT_SEC,
    VOLUME_MOUNT,
    _curl,
    _python,
    _setup_cache,
    _torchrun,
    image,
    secret,
    volume,
)


app = modal.App("nanochat-experiments")

ORIGINAL_DEPTH = 24
ORIGINAL_NUM_SHARDS = 170
ORIGINAL_DEVICE_BATCH_SIZE = 16
BASELINE_SFT_STEPS = 483
PRETRAIN_GPU = "H100:8"
FINETUNE_GPU = "H100:8"
ANALYSIS_GPU = "A10G:1"
PRETRAIN_NPROC = 8
FINETUNE_NPROC = 8
RL_TIMEOUT_SEC = 60 * 60 * 6
EXTRA_SFT_DATASETS = {
    "ultrachat_200k",
    "tulu_if",
    "tulu-3-sft-personas-instruction-following",
    "tulu_math",
    "tulu_code",
}


def _identity_path():
    return os.path.join(NANOCHAT_CACHE, "identity_conversations.jsonl")


def _extra_jsonl_path(dataset_alias: str) -> str:
    return os.path.join(BASE_DIR, "sft_extra_data", f"{dataset_alias}.jsonl")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=8,
    memory=16384,
    timeout=DOWNLOAD_TIMEOUT_SEC,
)
def stage_prepare_pretrain_data(num_shards: int = ORIGINAL_NUM_SHARDS) -> None:
    _setup_cache()
    _python("nanochat.dataset", [f"-n {num_shards}"])
    volume.commit()


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:1",
    timeout=60 * 30,
)
def stage_train_tokenizer() -> None:
    _setup_cache()
    _python("scripts.tok_train")
    _python("scripts.tok_eval")
    volume.commit()


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=PRETRAIN_GPU,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain_original(
    wandb_run: str,
    model_tag: str = "",
    depth: int = ORIGINAL_DEPTH,
    device_batch_size: int = ORIGINAL_DEVICE_BATCH_SIZE,
) -> None:
    _setup_cache()
    _python("nanochat.report", ["reset"])
    args = [
        f"--depth={depth}",
        "--target-param-data-ratio=9.5",
        f"--device-batch-size={device_batch_size}",
        "--fp8",
        f"--run={wandb_run}",
    ]
    if model_tag:
        args.append(f"--model-tag={model_tag}")
    _torchrun("scripts.base_train", args, nproc=PRETRAIN_NPROC)
    volume.commit()


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=PRETRAIN_GPU,
    timeout=60 * 60 * 2,
)
def stage_post_pretrain_eval(model_tag: str = "", step: int = -1) -> None:
    _setup_cache()
    eval_args = ["--device-batch-size", str(ORIGINAL_DEVICE_BATCH_SIZE)]
    if model_tag:
        eval_args.extend(["--model-tag", model_tag])
    if step >= 0:
        eval_args.extend(["--step", str(step)])
    _torchrun("scripts.base_eval", eval_args, nproc=PRETRAIN_NPROC)
    volume.commit()


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=ANALYSIS_GPU,
    timeout=60 * 60,
)
def stage_base_chat_eval(
    wandb_run: str,
    model_tag: str = "",
    step: int = -1,
    batch_size: int = 8,
    max_cat: int = -1,
    max_sample: int = 24,
) -> None:
    _setup_cache()
    args = [
        f"--run={wandb_run}",
        "--project=nanochat-sft",
        f"--batch-size={batch_size}",
        f"--max-cat={max_cat}",
        f"--max-sample={max_sample}",
    ]
    if model_tag:
        args.append(f"--model-tag={model_tag}")
    if step >= 0:
        args.append(f"--step={step}")
    _python("scripts.base_chat_eval", args)
    volume.commit()


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=8,
    memory=16384,
    timeout=60 * 60,
)
def stage_prepare_extra_sft_dataset(dataset: str, max_rows: int = -1) -> str:
    if dataset not in EXTRA_SFT_DATASETS:
        raise ValueError(f"Unknown dataset alias: {dataset}")
    _setup_cache()
    args = [f"--dataset={dataset}"]
    if max_rows > 0:
        args.append(f"--max-rows={max_rows}")
    _python("assignments.a4.prepare_sft_dataset", args)
    volume.commit()
    return _extra_jsonl_path(dataset)


def _run_sft(
    wandb_run: str,
    model_tag: str = "",
    model_step: int = -1,
    output_tag: str = "",
    extra_jsonls: list[str] | None = None,
    num_iterations: int = -1,
) -> None:
    _curl(IDENTITY_JSONL_URL, _identity_path())
    args = [
        f"--run={wandb_run}",
        f"--device-batch-size={ORIGINAL_DEVICE_BATCH_SIZE}",
    ]
    if model_tag:
        args.append(f"--model-tag={model_tag}")
    if model_step >= 0:
        args.append(f"--model-step={model_step}")
    if output_tag:
        args.append(f"--output-tag={output_tag}")
    if num_iterations > 0:
        args.append(f"--num-iterations={num_iterations}")
    for path in extra_jsonls or []:
        args.append(f"--extra-jsonl={path}")
    _torchrun("scripts.chat_sft", args, nproc=FINETUNE_NPROC)


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=FINETUNE_GPU,
    timeout=FINETUNE_TIMEOUT_SEC,
)
def stage_sft_original(
    wandb_run: str,
    model_tag: str = "",
    model_step: int = -1,
    output_tag: str = "",
    num_iterations: int = -1,
) -> None:
    _setup_cache()
    _run_sft(
        wandb_run=wandb_run,
        model_tag=model_tag,
        model_step=model_step,
        output_tag=output_tag,
        num_iterations=num_iterations,
    )
    eval_args = ["-i", "sft"]
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
    gpu=FINETUNE_GPU,
    timeout=FINETUNE_TIMEOUT_SEC,
)
def stage_sft_extra(
    dataset: str,
    wandb_run: str,
    model_tag: str = "",
    model_step: int = -1,
    output_tag: str = "",
    num_iterations: int = BASELINE_SFT_STEPS,
) -> None:
    if dataset not in EXTRA_SFT_DATASETS:
        raise ValueError(f"Unknown dataset alias: {dataset}")
    _setup_cache()
    extra_path = _extra_jsonl_path(dataset)
    if not os.path.exists(extra_path):
        raise FileNotFoundError(f"Missing prepared dataset {extra_path}. Run stage_prepare_extra_sft_dataset first.")
    _run_sft(
        wandb_run=wandb_run,
        model_tag=model_tag,
        model_step=model_step,
        output_tag=output_tag,
        extra_jsonls=[extra_path],
        num_iterations=num_iterations,
    )
    eval_args = ["-i", "sft"]
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
    gpu=FINETUNE_GPU,
    timeout=RL_TIMEOUT_SEC,
)
def stage_rl_gsm8k(
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
    args = [f"--run={wandb_run}", f"--source={source}", f"--wandb-resume={wandb_resume}"]
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
    _torchrun("scripts.chat_rl", args, nproc=FINETUNE_NPROC)
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
    cpu=2,
    memory=4096,
    timeout=60 * 30,
)
def stage_generate_report() -> None:
    _setup_cache()
    _python("nanochat.report", ["generate"])
    volume.commit()


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=2,
    memory=4096,
    timeout=60 * 30,
)
def stage_compare_wandb(
    run_a: str,
    run_b: str,
    metric: str,
    label_a: str = "",
    label_b: str = "",
    smooth_window: int = 1,
) -> None:
    _setup_cache()
    args = [
        f"--run={run_a}",
        f"--run={run_b}",
        f"--metric={metric}",
        f"--smooth-window={smooth_window}",
    ]
    if label_a:
        args.append(f"--label={label_a}")
    if label_b:
        args.append(f"--label={label_b}")
    _python("assignments.a4.compare_wandb_runs", args)
    volume.commit()


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=ANALYSIS_GPU,
    timeout=60 * 60,
)
def stage_gsm8k_error_analysis(
    source: str,
    split: str = "test",
    model_tag: str = "",
    step: int = -1,
    max_examples: int = 200,
    output_dir: str = "",
    max_new_tokens: int = -1,
    progress_every: int = 10,
    log_example_every: int = 1,
) -> None:
    _setup_cache()
    args = [
        f"--source={source}",
        f"--split={split}",
        f"--max-examples={max_examples}",
        f"--progress-every={progress_every}",
        f"--log-example-every={log_example_every}",
    ]
    if model_tag:
        args.append(f"--model-tag={model_tag}")
    if step >= 0:
        args.append(f"--step={step}")
    if output_dir:
        args.append(f"--output-dir={output_dir}")
    if max_new_tokens > 0:
        args.append(f"--max-new-tokens={max_new_tokens}")
    _python("assignments.a4.gsm8k_error_analysis", args)
    volume.commit()


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=2,
    memory=4096,
    timeout=60 * 30,
)
def stage_compare_gsm8k_analyses(
    csv_paths: list[str],
    labels: list[str],
    output_dir: str,
    top_disagreements: int = 20,
) -> None:
    _setup_cache()
    args = [f"--output-dir={output_dir}", f"--top-disagreements={top_disagreements}"]
    for path in csv_paths:
        args.append(f"--csv={path}")
    for label in labels:
        args.append(f"--label={label}")
    _python("assignments.a4.compare_gsm8k_analyses", args)
    volume.commit()

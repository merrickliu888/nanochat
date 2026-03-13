"""
Reinforcement learning on GSM8K with alternative reward systems.

This script is intentionally separate from scripts/chat_rl.py so that reward
ablations do not modify the baseline RL implementation.
"""

import argparse
import os

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import (
    find_last_step,
    find_largest_model,
    load_model,
    load_optimizer_state,
    save_checkpoint,
)
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K
from assignments.a4.gsm8k_reward_systems import compute_reward


parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K with reward-system ablations")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--wandb-id", type=str, default=None, help="W&B run id to resume into")
parser.add_argument("--wandb-resume", type=str, default="never", choices=["allow", "never", "must", "auto"], help="W&B resume mode")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--source", type=str, default="sft", choices=["sft", "rl"], help="checkpoint source to load from")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--output-tag", type=str, default=None, help="model tag to save RL checkpoints under")
parser.add_argument("--reward-system", type=str, default="baseline", choices=["baseline", "format_aware", "accuracy_shaped", "combined"], help="reward system to optimize")
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
args = parser.parse_args()
user_config = vars(args).copy()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_entity = os.environ.get("WANDB_ENTITY") or None
wandb_kwargs = {"project": "nanochat-rl-rewards", "name": args.run, "config": user_config}
if wandb_entity:
    wandb_kwargs["entity"] = wandb_entity
if args.wandb_id:
    wandb_kwargs["id"] = args.wandb_id
    wandb_kwargs["resume"] = args.wandb_resume
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(**wandb_kwargs)

resolved_model_tag = args.model_tag
resolved_model_step = args.model_step
if args.source == "rl":
    rl_root = os.path.join(get_base_dir(), "chatrl_checkpoints")
    if resolved_model_tag is None:
        resolved_model_tag = find_largest_model(rl_root)
        print0(f"No RL model tag provided, using latest RL tag: {resolved_model_tag}")
    rl_checkpoint_dir = os.path.join(rl_root, resolved_model_tag)
    if resolved_model_step is None:
        resolved_model_step = find_last_step(rl_checkpoint_dir)
        print0(f"No RL model step provided, using latest RL step: {resolved_model_step}")

model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=resolved_model_tag, step=resolved_model_step)
engine = Engine(model, tokenizer)

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")
rank_indices = list(range(ddp_rank, len(train_task), ddp_world_size))


@torch.no_grad()
def get_batch(example_idx: int, current_step: int):
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    conversation = train_task[example_idx]
    tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(tokens)

    model.eval()
    generated_token_sequences = []
    masks = []
    reward_values = []
    component_rows = []
    num_sampling_steps = args.num_samples // args.device_batch_size
    for sampling_step in range(num_sampling_steps):
        seed = hash((current_step, example_idx, sampling_step)) & 0x7FFFFFFF
        generated_batch, masks_batch = engine.generate_batch(
            tokens,
            num_samples=args.device_batch_size,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=seed,
        )
        generated_token_sequences.extend(generated_batch)
        masks.extend(masks_batch)

    for sample_tokens in generated_token_sequences:
        generated_tokens = sample_tokens[prefix_length:]
        generated_text = tokenizer.decode(generated_tokens)
        reward, components = compute_reward(train_task, conversation, generated_text, args.reward_system)
        reward_values.append(reward)
        component_rows.append(components)

    max_length = max(len(seq) for seq in generated_token_sequences)
    padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
    padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
    ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
    mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[mask_ids[:, 1:] == 0] = -1
    rewards = torch.tensor(reward_values, dtype=torch.float, device=device)
    advantages = rewards - rewards.mean()
    return generated_token_sequences, inputs, targets, rewards, advantages, component_rows


def run_gsm8k_eval(task, tokenizer, engine, max_examples=None, num_samples=1, max_completion_tokens=256, temperature=0.0, top_k=50):
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        assert num_samples <= args.device_batch_size
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        yield {"idx": idx, "outcomes": outcomes}


optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

optimizer_state = None
if args.source == "rl":
    optimizer_state = load_optimizer_state("rl", device, rank=ddp_rank, model_tag=resolved_model_tag, step=resolved_model_step)

if optimizer_state is not None:
    optimizer.load_state_dict(optimizer_state)
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
else:
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]
    if args.source == "rl":
        print0("Optimizer state was not found for the RL checkpoint; resuming with a fresh optimizer state.")


def get_lr_multiplier(it):
    return 1.0 - it / num_steps


print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}")
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size
print0(f"Calculated examples per rank: {examples_per_rank}")

start_step = 0
if args.source == "rl":
    assert resolved_model_step is not None
    start_step = resolved_model_step + 1
    print0(f"Resuming RL training from step {resolved_model_step}; next optimization step is {start_step}")

for step in range(start_step, num_steps):
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device)
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0)
        records = list(records_iter)
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item()
        wandb_run.log({"step": step, **{f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}})
        print0(f"Step {step} | " + ", ".join(f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)))

    rewards_list = []
    sequence_lengths = []
    component_totals = {}
    component_count = 0
    for example_step in range(examples_per_rank):
        flat_example_idx = step * examples_per_rank + example_step
        example_idx = rank_indices[flat_example_idx % len(rank_indices)]
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all, component_rows = get_batch(example_idx, step)
        model.train()
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            advantages = advantages_all[b0:b1]
            rewards = rewards_all[b0:b1]
            logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)
        for component in component_rows:
            for name, value in component.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value)
            component_count += 1

    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")

    component_logs = {}
    for name, total in component_totals.items():
        component_logs[f"reward_component/{name}"] = total / max(component_count, 1)
    wandb_run.log({"step": step, "reward": mean_reward, "sequence_length": mean_sequence_length, **component_logs})

    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({"step": step, "lrm": lrm})

    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        reward_suffix = args.reward_system.replace("_", "-")
        default_tag = f"{args.model_tag}-{reward_suffix}" if args.model_tag else f"d{depth}-{reward_suffix}"
        output_dirname = args.output_tag if args.output_tag else default_tag
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "model_config": model.config.__dict__,
                "user_config": user_config,
            },
        )
        print0(f"Saved reward-ablation checkpoint to {checkpoint_dir}")

from nanochat.report import get_report

get_report().log(section="Chat RL Reward Ablation", data=[user_config])

wandb_run.finish()
compute_cleanup()

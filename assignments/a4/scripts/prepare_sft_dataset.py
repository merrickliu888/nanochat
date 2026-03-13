"""
Prepare additional SFT datasets for nanochat.

The output format matches tasks.customjson.CustomJSON:
each line is a JSON array of alternating user/assistant messages.
"""

import argparse
import json
import os
from collections.abc import Iterable


DATASETS = {
    "ultrachat_200k": {
        "path": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "messages_field": "messages",
        "description": "General multi-turn assistant conversations; useful as a higher-volume conversational complement to SmolTalk.",
    },
    "tulu_if": {
        "path": "allenai/tulu-3-sft-personas-instruction-following",
        "split": "train",
        "messages_field": "messages",
        "description": "Instruction-following dialogues with explicit persona/control signals; useful for instruction adherence.",
    },
    "tulu-3-sft-personas-instruction-following": {
        "path": "allenai/tulu-3-sft-personas-instruction-following",
        "split": "train",
        "messages_field": "messages",
        "description": "Instruction-following dialogues with explicit persona/control signals; useful for instruction adherence.",
    },
    "tulu_math": {
        "path": "allenai/tulu-3-sft-personas-math",
        "split": "train",
        "messages_field": "messages",
        "description": "Math-focused supervised traces; useful as a targeted supplement for GSM8K-like reasoning.",
    },
    "tulu_code": {
        "path": "allenai/tulu-3-sft-personas-code",
        "split": "train",
        "messages_field": "messages",
        "description": "Code-focused supervised traces; useful as a targeted supplement for HumanEval-style generation.",
    },
}


def flatten_content(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"].strip()
        if "content" in content:
            return flatten_content(content["content"])
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, Iterable):
        parts = []
        for item in content:
            text = flatten_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def normalize_role(role):
    role = role.lower()
    if role in {"user", "human"}:
        return "user"
    if role in {"assistant", "gpt", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return None


def sanitize_messages(messages):
    cleaned = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(str(message.get("role", "")))
        if role is None or role == "system":
            continue
        text = flatten_content(message.get("content", ""))
        if not text:
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n\n" + text
        else:
            cleaned.append({"role": role, "content": text})

    if not cleaned:
        return None
    if cleaned[0]["role"] != "user":
        return None
    if len(cleaned) < 2:
        return None
    if cleaned[-1]["role"] != "assistant":
        cleaned = cleaned[:-1]
    if len(cleaned) < 2 or len(cleaned) % 2 != 0:
        return None
    for idx, message in enumerate(cleaned):
        expected = "user" if idx % 2 == 0 else "assistant"
        if message["role"] != expected:
            return None
    return cleaned


def main():
    from datasets import load_dataset

    from nanochat.common import get_base_dir

    parser = argparse.ArgumentParser(description="Prepare an additional SFT dataset as nanochat CustomJSON JSONL")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()), help="dataset alias to prepare")
    parser.add_argument("--max-rows", type=int, default=-1, help="max rows to export (-1 = all)")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed before slicing")
    parser.add_argument("--output", type=str, default=None, help="output JSONL path (default: $NANOCHAT_BASE_DIR/sft_extra_data/<alias>.jsonl)")
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    ds = load_dataset(spec["path"], split=spec["split"]).shuffle(seed=args.seed)

    if args.max_rows > 0:
        ds = ds.select(range(min(args.max_rows, len(ds))))

    base_dir = get_base_dir()
    default_dir = os.path.join(base_dir, "sft_extra_data")
    os.makedirs(default_dir, exist_ok=True)
    output_path = args.output or os.path.join(default_dir, f"{args.dataset}.jsonl")

    kept = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            messages = row.get(spec["messages_field"])
            if messages is None:
                skipped += 1
                continue
            cleaned = sanitize_messages(messages)
            if cleaned is None:
                skipped += 1
                continue
            f.write(json.dumps(cleaned, ensure_ascii=True) + "\n")
            kept += 1

    print(f"Prepared dataset: {args.dataset}")
    print(f"Source: {spec['path']} [{spec['split']}]")
    print(f"Description: {spec['description']}")
    print(f"Output: {output_path}")
    print(f"Kept rows: {kept}")
    print(f"Skipped rows: {skipped}")


if __name__ == "__main__":
    main()

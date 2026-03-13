"""
Reward systems for GSM8K RL experiments.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from tasks.gsm8k import extract_answer


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
FINAL_MARKER_RE = re.compile(r"####\s*-?\d[\d,]*(?:\.\d+)?")


def parse_decimal(text):
    if text is None:
        return None
    try:
        return Decimal(text.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def has_any_number(text: str) -> bool:
    return bool(NUMBER_RE.search(text))


def has_final_marker(text: str) -> bool:
    return bool(FINAL_MARKER_RE.search(text))


def has_worked_solution(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        return True
    if "<<" in text or ">>" in text:
        return True
    if "=" in text:
        return True
    return False


def near_match(pred_num, gold_num) -> bool:
    pred_val = parse_decimal(pred_num)
    gold_val = parse_decimal(gold_num)
    if pred_val is None or gold_val is None:
        return False
    if pred_val == gold_val:
        return True
    if pred_val == gold_val + 1 or pred_val == gold_val - 1:
        return True
    if gold_val != 0:
        rel_err = abs(pred_val - gold_val) / abs(gold_val)
        if rel_err <= Decimal("0.10"):
            return True
    return False


def reward_components(task, conversation, assistant_response: str):
    completion = assistant_response.strip()
    gold_text = conversation["messages"][-1]["content"][-1]["text"]
    gold_num = extract_answer(gold_text)
    pred_num = extract_answer(assistant_response)
    exact = float(task.evaluate(conversation, assistant_response))

    return {
        "nonempty": float(bool(completion)),
        "has_number": float(has_any_number(assistant_response)),
        "has_final_marker": float(has_final_marker(assistant_response)),
        "worked_solution": float(has_worked_solution(assistant_response)),
        "near_match": float(near_match(pred_num, gold_num) and not exact),
        "exact_match": exact,
    }


def compute_reward(task, conversation, assistant_response: str, reward_system: str):
    components = reward_components(task, conversation, assistant_response)

    if reward_system == "baseline":
        reward = components["exact_match"]
    elif reward_system == "format_aware":
        reward = (
            0.05 * components["nonempty"]
            + 0.15 * components["has_number"]
            + 0.20 * components["has_final_marker"]
            + 0.60 * components["exact_match"]
        )
    elif reward_system == "accuracy_shaped":
        reward = (
            0.05 * components["nonempty"]
            + 0.10 * components["has_number"]
            + 0.15 * components["has_final_marker"]
            + 0.20 * components["worked_solution"]
            + 0.20 * components["near_match"]
            + 0.30 * components["exact_match"]
        )
    elif reward_system == "combined":
        reward = (
            0.05 * components["nonempty"]
            + 0.10 * components["has_number"]
            + 0.15 * components["has_final_marker"]
            + 0.15 * components["worked_solution"]
            + 0.15 * components["near_match"]
            + 0.40 * components["exact_match"]
        )
    else:
        raise ValueError(f"Unknown reward system: {reward_system}")

    components["reward_total"] = float(reward)
    return float(reward), components

#!/usr/bin/env python3
"""Offline Core-LT TCAD margin diagnostic.

This script samples TCAD-eligible LIBERO-Core-LT transitions, computes the
average action-token log-probability under the correct instruction and a
manifest-valid negative instruction, and summarizes the margin:

    margin = log p(action | image, correct instruction)
           - log p(action | image, negative instruction)

It is designed as a diagnostic, not as a training component. It can compare
BC/RBTAD/Rare-BC/APA checkpoints and optionally correlate per-task margin
changes with per-task evaluation success changes from
`core_lt_priority_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NO_GCE_CHECK", "true")

import numpy as np
import tensorflow_datasets as tfds
import torch
from PIL import Image

from prismatic.models.load import load_vla
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ACTION_TOKENIZERS
from prismatic.vla.datasets.datasets import RLDSBatchTransform

CORE_TASK_COUNTS = {
    "pick up the black bowl next to the plate and place it on the plate": 46,
    "pick up the black bowl next to the cookie box and place it on the plate": 28,
    "pick up the black bowl on the cookie box and place it on the plate": 19,
    "pick up the ketchup and place it in the basket": 15,
    "pick up the alphabet soup and place it in the basket": 11,
    "push the plate to the front of the stove": 9,
    "put the bowl on top of the cabinet": 8,
    "put the cream cheese in the bowl": 7,
    "put the wine bottle on top of the cabinet": 6,
    "put the wine bottle on the rack": 5,
}

CORE_VALID_NEGATIVES = {
    "put the bowl on top of the cabinet": "put the wine bottle on top of the cabinet",
    "put the cream cheese in the bowl": "pick up the ketchup and place it in the basket",
    "put the wine bottle on top of the cabinet": "put the wine bottle on the rack",
    "put the wine bottle on the rack": "put the wine bottle on top of the cabinet",
}

TASK_ORDER = list(CORE_TASK_COUNTS)
TASK_ID = {instruction: idx for idx, instruction in enumerate(TASK_ORDER)}


def stable_negative(instruction: str) -> str | None:
    instruction = instruction.lower().strip()
    if instruction in CORE_VALID_NEGATIVES:
        return CORE_VALID_NEGATIVES[instruction]
    candidates = [item for item in TASK_ORDER if item != instruction]
    if not candidates:
        return None
    # Deterministic fallback mirrors the dataset-level manifest fallback spirit,
    # but explicit task ids make the diagnostic reproducible outside the dataset transform.
    digest = hashlib.sha1(("libero_core_lt::" + instruction).encode("utf-8")).hexdigest()
    return candidates[int(digest, 16) % len(candidates)]


def image_from_step(step: dict[str, Any]) -> Image.Image:
    obs = step["observation"]
    if "image" in obs:
        return Image.fromarray(obs["image"])
    if "image_primary" in obs:
        return Image.fromarray(obs["image_primary"])
    raise KeyError(f"No image field in observation keys={list(obs)}")


def make_batch_transform(vla, action_tokenizer_name: str):
    tokenizer = vla.llm_backbone.get_tokenizer()
    action_tokenizer = ACTION_TOKENIZERS[action_tokenizer_name](tokenizer)
    transform = RLDSBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=tokenizer,
        image_transform=vla.vision_backbone.get_image_transform(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        predict_stop_token=True,
        image_window_size=1,
        use_wrist_image=False,
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        padding_side="right",
    )
    return transform, collator, action_tokenizer


def build_sample(transform, image: Image.Image, action: np.ndarray, instruction: str):
    rlds_batch = {
        "dataset_name": "libero_core_lt",
        "action": np.asarray(action, dtype=np.float32)[None, :],
        "task": {"language_instruction": instruction.encode("utf-8")},
        "observation": {"image_primary": np.expand_dims(np.asarray(image), axis=0)},
    }
    return transform(rlds_batch)


def to_device(value, dtype):
    if torch.is_tensor(value):
        value = value.cuda(non_blocking=True)
        if value.is_floating_point():
            value = value.to(dtype=dtype)
        return value
    if isinstance(value, dict):
        return {k: to_device(v, dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, dtype) for v in value]
    if isinstance(value, tuple):
        return tuple(to_device(v, dtype) for v in value)
    return value


def action_logprob(vla, batch, action_tokenizer, token_scope: str):
    vision_dtype = vla.vision_backbone.half_precision_dtype
    with torch.no_grad():
        output = vla(
            input_ids=batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            pixel_values=to_device(batch["pixel_values"], vision_dtype),
            labels=batch["labels"].cuda(),
        )
        logits = output.logits[:, vla.vision_backbone.num_patches : -1].float()
        labels = batch["labels"][:, 1:].cuda()
        mask = (action_tokenizer.action_token_end_idx > labels) & (
            labels > action_tokenizer.action_token_begin_idx
        )
        if token_scope.strip().lower() in {"last", "last_action", "gripper_phase"}:
            positions = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
            last_positions = positions.masked_fill(~mask, -1).max(dim=1).values
            mask = mask & (positions == last_positions.view(-1, 1))
        safe_labels = labels.clamp_min(0)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        selected = selected.masked_fill(~mask, 0.0)
        counts = mask.sum(dim=1).clamp_min(1)
        return (selected.sum(dim=1) / counts).detach().cpu().numpy(), counts.detach().cpu().numpy()


def sample_core_lt(tfds_dir: Path, max_samples_per_task: int, tail_max_count: int, max_episodes: int):
    builder = tfds.builder_from_directory(str(tfds_dir))
    dataset = builder.as_dataset(split="train")
    rows = []
    counts = defaultdict(int)
    skipped = defaultdict(int)
    for episode_idx, episode in enumerate(dataset.take(max_episodes)):
        steps = list(episode["steps"].as_numpy_iterator())
        for step_idx, step in enumerate(steps):
            instruction = step["language_instruction"].decode("utf-8").lower().strip()
            task_count = CORE_TASK_COUNTS.get(instruction, 10**9)
            if tail_max_count > 0 and task_count > tail_max_count:
                skipped["not_tail"] += 1
                continue
            if counts[instruction] >= max_samples_per_task:
                skipped["task_quota"] += 1
                continue
            action = np.asarray(step["action"], dtype=np.float32)
            if float(action[-1]) <= 0:
                skipped["not_gripper_positive"] += 1
                continue
            wrong = stable_negative(instruction)
            if wrong is None:
                skipped["no_negative"] += 1
                continue
            rows.append(
                {
                    "episode_idx": int(episode_idx),
                    "step_idx": int(step_idx),
                    "task_id": TASK_ID.get(instruction, -1),
                    "instruction": instruction,
                    "negative_instruction": wrong,
                    "task_count": int(task_count),
                    "action": action,
                    "image": image_from_step(step),
                }
            )
            counts[instruction] += 1
        if all(counts[t] >= max_samples_per_task for t, c in CORE_TASK_COUNTS.items() if tail_max_count <= 0 or c <= tail_max_count):
            break
    serializable = []
    for row in rows:
        serializable.append({k: v for k, v in row.items() if k not in {"action", "image"}})
    return rows, dict(skipped), serializable


def load_model(checkpoint: str):
    vla = load_vla(
        checkpoint,
        hf_token="",
        load_for_training=False,
        image_sequence_len=1,
        instruction_formatting=False,
    )
    dtype = vla.llm_backbone.half_precision_dtype
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=dtype)
    vla.to(dtype=dtype)
    vla.cuda()
    vla.eval()
    return vla


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "positive_rate": None, "p10": None, "p90": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "positive_rate": float((arr > 0).mean()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(xa.std()) == 0.0 or float(ya.std()) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return pearson(rankdata(x), rankdata(y))


def load_eval_summary(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def task_success(summary: dict, key: str) -> dict[int, float]:
    entry = summary.get("entries", {}).get(key, {})
    values = entry.get("per_task") or []
    return {i: float(v) for i, v in enumerate(values)}


def build_correlations(model_summaries: dict, eval_summary: dict | None, pairs: list[tuple[str, str, str, str]]):
    if not eval_summary:
        return []
    out = []
    for label, margin_a, margin_b, eval_pair in pairs:
        if margin_a not in model_summaries or margin_b not in model_summaries:
            continue
        if eval_pair == "rbtad_vs_legacy_bc":
            succ_a = task_success(eval_summary, "bc_seed7_legacy")
            succ_b = task_success(eval_summary, "rbtad_seed7")
        elif eval_pair == "rbtad_vs_matched_bc":
            succ_a = task_success(eval_summary, "matched_bc_seed7")
            succ_b = task_success(eval_summary, "rbtad_seed7")
        else:
            continue
        xs, ys, rows = [], [], []
        tasks = sorted(set(model_summaries[margin_a]["by_task"]) & set(model_summaries[margin_b]["by_task"]))
        for task_id in tasks:
            if task_id not in succ_a or task_id not in succ_b:
                continue
            ma = model_summaries[margin_a]["by_task"][task_id]["mean"]
            mb = model_summaries[margin_b]["by_task"][task_id]["mean"]
            if ma is None or mb is None:
                continue
            margin_delta = float(mb - ma)
            success_delta = float(succ_b[task_id] - succ_a[task_id])
            xs.append(margin_delta)
            ys.append(success_delta)
            rows.append({"task_id": task_id, "margin_delta": margin_delta, "success_delta": success_delta})
        out.append({
            "label": label,
            "n_tasks": len(rows),
            "pearson": pearson(xs, ys),
            "spearman": spearman(xs, ys),
            "rows": rows,
        })
    return out


def parse_model_arg(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise argparse.ArgumentTypeError("model must be label=/path/to/checkpoint")
    label, checkpoint = item.split("=", 1)
    return label.strip(), checkpoint.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfds-dir", type=Path, default=Path("/mnt/data/cyh/tensorflow_datasets/libero_core_lt/1.0.0"))
    parser.add_argument("--model", action="append", type=parse_model_arg, required=True, help="label=/path/to/checkpoint")
    parser.add_argument("--summary-json", type=Path, default=Path("/mnt/data/cyh/VLA-long-tail/autoresearch/state/core_lt_priority_summary.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples-per-task", type=int, default=40)
    parser.add_argument("--max-episodes", type=int, default=160)
    parser.add_argument("--tail-max-count", type=int, default=9)
    parser.add_argument("--action-tokenizer", default="extra_action_tokenizer")
    parser.add_argument("--token-scope", choices=["all", "last_action"], default="all")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    samples, skipped, sample_meta = sample_core_lt(
        args.tfds_dir,
        max_samples_per_task=args.max_samples_per_task,
        tail_max_count=args.tail_max_count,
        max_episodes=args.max_episodes,
    )
    if not samples:
        raise RuntimeError("No diagnostic samples collected")

    all_model_rows: dict[str, list[dict[str, Any]]] = {}
    model_summaries: dict[str, dict[str, Any]] = {}
    for label, checkpoint in args.model:
        vla = load_model(checkpoint)
        transform, collator, action_tokenizer = make_batch_transform(vla, args.action_tokenizer)
        rows = []
        by_task_values: dict[int, list[float]] = defaultdict(list)
        for sample in samples:
            correct = build_sample(transform, sample["image"], sample["action"], sample["instruction"])
            wrong = build_sample(transform, sample["image"], sample["action"], sample["negative_instruction"])
            batch = collator([correct, wrong])
            scores, token_counts = action_logprob(vla, batch, action_tokenizer, args.token_scope)
            margin = float(scores[0] - scores[1])
            task_id = int(sample["task_id"])
            by_task_values[task_id].append(margin)
            rows.append({
                "episode_idx": sample["episode_idx"],
                "step_idx": sample["step_idx"],
                "task_id": task_id,
                "instruction": sample["instruction"],
                "negative_instruction": sample["negative_instruction"],
                "task_count": sample["task_count"],
                "score_correct": float(scores[0]),
                "score_negative": float(scores[1]),
                "margin": margin,
                "token_count": float(token_counts[0]),
            })
        all_model_rows[label] = rows
        model_summaries[label] = {
            "checkpoint": checkpoint,
            "overall": summarize([row["margin"] for row in rows]),
            "by_task": {task_id: summarize(vals) for task_id, vals in sorted(by_task_values.items())},
        }
        del vla
        torch.cuda.empty_cache()

    eval_summary = load_eval_summary(args.summary_json)
    correlations = build_correlations(
        model_summaries,
        eval_summary,
        pairs=[
            ("RBTAD margin delta vs legacy BC success delta", "bc_seed7", "rbtad_seed7", "rbtad_vs_legacy_bc"),
            ("RBTAD margin delta vs matched BC success delta", "matched_bc_seed7", "rbtad_seed7", "rbtad_vs_matched_bc"),
        ],
    )
    output = {
        "config": {
            "tfds_dir": str(args.tfds_dir),
            "tail_max_count": args.tail_max_count,
            "max_samples_per_task": args.max_samples_per_task,
            "max_episodes": args.max_episodes,
            "token_scope": args.token_scope,
            "summary_json": str(args.summary_json),
        },
        "sample_summary": {
            "num_samples": len(samples),
            "skipped": skipped,
            "samples": sample_meta,
        },
        "models": model_summaries,
        "correlations": correlations,
        "rows": all_model_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"sample_count": len(samples), "models": model_summaries, "correlations": correlations}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()



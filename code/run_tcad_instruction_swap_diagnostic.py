import argparse
import json
import os
import re
from pathlib import Path

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


OBJECTS = [
    "black bowl",
    "cookie box",
    "plate",
    "ketchup",
    "basket",
    "alphabet soup",
    "stove",
    "cabinet",
    "cream cheese",
    "wine bottle",
    "rack",
]


def extract_target_and_distractor(instruction: str):
    instruction = instruction.lower()
    patterns = [
        r"^pick up the (.*?) next to the (.*?) and place it on the (.*?)$",
        r"^pick up the (.*?) on the (.*?) and place it on the (.*?)$",
        r"^pick up the (.*?) and place it in the (.*?)$",
        r"^push the (.*?) to the front of the (.*?)$",
        r"^put the (.*?) on top of the (.*?)$",
        r"^put the (.*?) on the (.*?)$",
        r"^put the (.*?) in the (.*?)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, instruction)
        if match:
            groups = [g.strip() for g in match.groups()]
            target = groups[0]
            distractors = [g for g in groups[1:] if g and g != target]
            if distractors:
                return target, distractors[0]

    for obj in OBJECTS:
        if obj in instruction:
            for replacement in OBJECTS:
                if replacement != obj and replacement not in instruction:
                    return obj, replacement
    return None, None


def make_wrong_instruction(instruction: str):
    target, distractor = extract_target_and_distractor(instruction)
    if not target or not distractor:
        return None, None, None
    return instruction.replace(target, distractor, 1), target, distractor


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


def build_sample(transform, image, action, instruction):
    rlds_batch = {
        "dataset_name": "libero_core_lt",
        "action": np.asarray(action, dtype=np.float32)[None, :],
        "task": {"language_instruction": instruction.encode("utf-8")},
        "observation": {"image_primary": np.expand_dims(np.asarray(image), axis=0)},
    }
    return transform(rlds_batch)


def action_logprob(vla, batch, action_tokenizer):
    vision_dtype = vla.vision_backbone.half_precision_dtype

    def to_cuda(value):
        if torch.is_tensor(value):
            value = value.cuda()
            if value.is_floating_point():
                value = value.to(dtype=vision_dtype)
            return value
        if isinstance(value, dict):
            return {k: to_cuda(v) for k, v in value.items()}
        if isinstance(value, list):
            return [to_cuda(v) for v in value]
        if isinstance(value, tuple):
            return tuple(to_cuda(v) for v in value)
        return value

    with torch.no_grad():
        output = vla(
            input_ids=batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            pixel_values=to_cuda(batch["pixel_values"]),
            labels=batch["labels"].cuda(),
        )
        logits = output.logits[:, vla.vision_backbone.num_patches : -1].float()
        labels = batch["labels"][:, 1:].cuda()
        mask = (action_tokenizer.action_token_end_idx > labels) & (
            labels > action_tokenizer.action_token_begin_idx
        )
        safe_labels = labels.clamp_min(0)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs.masked_fill(~mask, 0.0)
        counts = mask.sum(dim=1).clamp_min(1)
        return (token_log_probs.sum(dim=1) / counts).detach().cpu().numpy()


def find_precontact_indices(actions, states, k=25, backoff=3):
    actions = np.asarray(actions)
    close_candidates = np.where(actions[:, -1] > 0)[0]
    if len(close_candidates) == 0:
        return []
    close_step = int(close_candidates[0])
    start = max(0, close_step - k)
    end = max(start, close_step - backoff)
    return list(range(start, end))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tfds-dir", default="/mnt/data/cyh/tensorflow_datasets/libero_core_lt/1.0.0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-episodes", type=int, default=120)
    parser.add_argument("--action-tokenizer", default="extra_action_tokenizer")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")

    vla = load_vla(
        args.checkpoint,
        hf_token="",
        load_for_training=False,
        image_sequence_len=1,
        instruction_formatting=False,
    )
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(device)
    vla.eval()

    transform, collator, action_tokenizer = make_batch_transform(vla, args.action_tokenizer)

    builder = tfds.builder_from_directory(args.tfds_dir)
    dataset = builder.as_dataset(split="train")

    rows = []
    skipped = {"no_wrong_instruction": 0, "no_precontact": 0}

    for episode_idx, episode in enumerate(dataset.take(args.max_episodes)):
        steps = list(episode["steps"].as_numpy_iterator())
        actions = np.stack([step["action"] for step in steps])
        states = np.stack([step["observation"]["state"] for step in steps])
        indices = find_precontact_indices(actions, states)
        if not indices:
            skipped["no_precontact"] += 1
            continue
        for step_idx in indices[:: max(1, len(indices) // 3)]:
            step = steps[step_idx]
            instruction = step["language_instruction"].decode("utf-8").lower()
            wrong_instruction, target, distractor = make_wrong_instruction(instruction)
            if wrong_instruction is None:
                skipped["no_wrong_instruction"] += 1
                continue

            image = Image.fromarray(step["observation"]["image"])
            action = step["action"]
            correct_sample = build_sample(transform, image, action, instruction)
            wrong_sample = build_sample(transform, image, action, wrong_instruction)
            batch = collator([correct_sample, wrong_sample])
            scores = action_logprob(vla, batch, action_tokenizer)
            margin = float(scores[0] - scores[1])
            rows.append(
                {
                    "episode_idx": episode_idx,
                    "step_idx": int(step_idx),
                    "instruction": instruction,
                    "wrong_instruction": wrong_instruction,
                    "target": target,
                    "distractor": distractor,
                    "score_correct": float(scores[0]),
                    "score_wrong": float(scores[1]),
                    "margin": margin,
                }
            )
            if len(rows) >= args.max_samples:
                break
        if len(rows) >= args.max_samples:
            break

    margins = np.array([row["margin"] for row in rows], dtype=np.float32)
    summary = {
        "num_samples": len(rows),
        "skipped": skipped,
        "mean_margin": float(margins.mean()) if len(margins) else None,
        "median_margin": float(np.median(margins)) if len(margins) else None,
        "positive_margin_rate": float((margins > 0).mean()) if len(margins) else None,
        "p10_margin": float(np.percentile(margins, 10)) if len(margins) else None,
        "p90_margin": float(np.percentile(margins, 90)) if len(margins) else None,
    }
    output = {"summary": summary, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from pathlib import Path


ROOT = Path("/mnt/data/cyh/VLA-long-tail")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"pattern not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1))


def patch_oxe() -> None:
    cfg = ROOT / "prismatic/vla/datasets/rlds/oxe/configs.py"
    mix = ROOT / "prismatic/vla/datasets/rlds/oxe/mixtures.py"
    trans = ROOT / "prismatic/vla/datasets/rlds/oxe/transforms.py"

    core_cfg = '''    "libero_core_lt": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
'''
    spatial_cfg = core_cfg + '''    "libero_spatial_lt": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
'''
    replace_once(cfg, core_cfg, spatial_cfg)

    core_mix = '''    "libero_core_lt": [
        ("libero_core_lt", 1.0),
    ],
'''
    spatial_mix = core_mix + '''    "libero_spatial_lt": [
        ("libero_spatial_lt", 1.0),
    ],
'''
    replace_once(mix, core_mix, spatial_mix)

    core_transform = '''    "libero_core_lt": libero_dataset_transform,
'''
    spatial_transform = core_transform + '''    "libero_spatial_lt": libero_dataset_transform,
'''
    replace_once(trans, core_transform, spatial_transform)


DATASET_MANIFEST_BLOCK = r'''TCAD_DATASET_MANIFESTS = {
    "libero_core_lt": {
        "task_counts": {
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
        },
        "valid_negatives": {
            "put the bowl on top of the cabinet": "put the wine bottle on top of the cabinet",
            "put the cream cheese in the bowl": "pick up the ketchup and place it in the basket",
            "put the wine bottle on top of the cabinet": "put the wine bottle on the rack",
            "put the wine bottle on the rack": "put the wine bottle on top of the cabinet",
        },
    },
    "libero_spatial_lt": {
        "task_counts": {
            "pick up the black bowl between the plate and the ramekin and place it on the plate": 44,
            "pick up the black bowl from table center and place it on the plate": 28,
            "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": 19,
            "pick up the black bowl next to the cookie box and place it on the plate": 15,
            "pick up the black bowl next to the plate and place it on the plate": 11,
            "pick up the black bowl next to the ramekin and place it on the plate": 9,
            "pick up the black bowl on the cookie box and place it on the plate": 8,
            "pick up the black bowl on the ramekin and place it on the plate": 7,
            "pick up the black bowl on the stove and place it on the plate": 6,
            "pick up the black bowl on the wooden cabinet and place it on the plate": 5,
        },
    },
}

TCAD_DATASET_ALIASES = {
    "libero_core_full": "libero_core_lt",
    "libero_core_apa": "libero_core_lt",
}


def _tcad_normalize_dataset_name(dataset_name: Any) -> str:
    if isinstance(dataset_name, bytes):
        dataset_name = dataset_name.decode("utf-8")
    return TCAD_DATASET_ALIASES.get(str(dataset_name), str(dataset_name))


def _tcad_manifest(dataset_name: Any) -> Optional[Dict[str, Dict[str, Any]]]:
    return TCAD_DATASET_MANIFESTS.get(_tcad_normalize_dataset_name(dataset_name))


def _tcad_task_count(instruction: str, dataset_name: Any) -> int:
    manifest = _tcad_manifest(dataset_name)
    if manifest is None:
        return 10**9
    return int(manifest["task_counts"].get(instruction.lower(), 10**9))


def _tcad_manifest_negative(instruction: str, dataset_name: Any) -> Optional[str]:
    manifest = _tcad_manifest(dataset_name)
    if manifest is None:
        return None
    instruction = instruction.lower()
    explicit = manifest.get("valid_negatives", {}).get(instruction)
    if explicit is not None and explicit in manifest["task_counts"]:
        return explicit
    candidates = [item for item in manifest["task_counts"] if item != instruction]
    if not candidates:
        return None
    digest = hashlib.sha1(f"{_tcad_normalize_dataset_name(dataset_name)}::{instruction}".encode("utf-8")).hexdigest()
    return candidates[int(digest, 16) % len(candidates)]


def _tcad_make_wrong_instruction(instruction: str, dataset_name: Any = None):
    instruction = instruction.lower()
    mode = os.environ.get("TCAD_NEGATIVE_MODE", "manifest").strip().lower()
    if mode in {"manifest", "dataset", "in_dataset", "relation_anchor"}:
        manifest_negative = _tcad_manifest_negative(instruction, dataset_name)
        if manifest_negative is not None:
            return manifest_negative
        if mode != "object_swap":
            return None

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
        if not match:
            continue
        groups = [g.strip() for g in match.groups()]
        target = groups[0]
        distractors = [g for g in groups[1:] if g and g != target]
        if distractors:
            return instruction.replace(target, distractors[0], 1)
    return None
'''


def patch_datasets() -> None:
    path = ROOT / "prismatic/vla/datasets/datasets.py"
    text = path.read_text()
    text = text.replace("from dataclasses import dataclass\n", "from dataclasses import dataclass\nimport hashlib\n")
    text = text.replace("from typing import Any, Dict, Tuple, Type\n", "from typing import Any, Dict, Optional, Tuple, Type\n")
    start = text.index("TCAD_OBJECTS = [")
    end = text.index("\n\n@dataclass\nclass RLDSBatchTransform:")
    text = text[:start] + DATASET_MANIFEST_BLOCK + text[end:]
    text = text.replace("task_count = TCAD_TASK_COUNTS.get(lang, 10**9)", "task_count = _tcad_task_count(lang, dataset_name)")
    text = text.replace("wrong_lang = _tcad_make_wrong_instruction(lang)", "wrong_lang = _tcad_make_wrong_instruction(lang, dataset_name)")
    text = text.replace("tail_allowed = tail_max_count <= 0 or TCAD_TASK_COUNTS.get(lang, 10**9) <= tail_max_count", "tail_allowed = tail_max_count <= 0 or task_count <= tail_max_count")
    text = text.replace("sample_weight=torch.tensor(sample_weight, dtype=torch.float32),\n", "sample_weight=torch.tensor(sample_weight, dtype=torch.float32),\n            task_count=torch.tensor(task_count, dtype=torch.int64),\n")
    path.write_text(text)


def patch_data_utils() -> None:
    path = ROOT / "prismatic/util/data_utils.py"
    text = path.read_text()
    text = text.replace('''        sample_weights = (
            torch.stack([instance["sample_weight"] for instance in instances])
            if "sample_weight" in instances[0]
            else None
        )
''', '''        sample_weights = (
            torch.stack([instance["sample_weight"] for instance in instances])
            if "sample_weight" in instances[0]
            else None
        )
        task_counts = (
            torch.stack([instance["task_count"] for instance in instances])
            if "task_count" in instances[0]
            else None
        )
''')
    text = text.replace('''        if sample_weights is not None:
            output["sample_weights"] = sample_weights
        return output
''', '''        if sample_weights is not None:
            output["sample_weights"] = sample_weights
        if task_counts is not None:
            output["task_counts"] = task_counts
        return output
''')
    path.write_text(text)


def patch_base_strategy() -> None:
    path = ROOT / "prismatic/training/strategies/base_strategy.py"
    text = path.read_text()
    old = '''                        if metrics.global_step == 0:
                            f.write("step,candidate_count,active_count,batch_size,tcad_loss,anchor_l2_loss\\n")
                        value = "nan" if tcad_loss is None else f"{float(tcad_loss.detach().cpu()):.6f}"
                        anchor_value = (
                            "nan"
                            if anchor_l2_loss is None
                            else f"{float(anchor_l2_loss.detach().cpu()):.6f}"
                        )
                        batch_size = int(tcad_active.numel()) if tcad_active is not None else 0
                        f.write(
                            f"{metrics.global_step},{tcad_candidate_count},{tcad_active_count},"
                            f"{batch_size},{value},{anchor_value}\\n"
                        )
'''
    new = '''                        if metrics.global_step == 0:
                            f.write(
                                "step,candidate_count,active_count,batch_size,tail_hit_count,"
                                "weighted_count,mean_sample_weight,tcad_loss,anchor_l2_loss\\n"
                            )
                        value = "nan" if tcad_loss is None else f"{float(tcad_loss.detach().cpu()):.6f}"
                        anchor_value = (
                            "nan"
                            if anchor_l2_loss is None
                            else f"{float(anchor_l2_loss.detach().cpu()):.6f}"
                        )
                        batch_size = int(tcad_active.numel()) if tcad_active is not None else 0
                        task_counts = batch.get("task_counts", None)
                        tail_limit = int(os.environ.get("TCAD_TAIL_MAX_COUNT", "0") or "0")
                        if task_counts is not None and tail_limit > 0:
                            tail_hit_count = int((task_counts.to(output.logits.device) <= tail_limit).sum().item())
                        else:
                            tail_hit_count = 0
                        sample_weights = batch.get("sample_weights", None)
                        if sample_weights is not None:
                            weights = sample_weights.float()
                            weighted_count = int((weights != 1.0).sum().item())
                            mean_sample_weight = float(weights.mean().item())
                        else:
                            weighted_count = 0
                            mean_sample_weight = 1.0
                        f.write(
                            f"{metrics.global_step},{tcad_candidate_count},{tcad_active_count},"
                            f"{batch_size},{tail_hit_count},{weighted_count},{mean_sample_weight:.6f},"
                            f"{value},{anchor_value}\\n"
                        )
'''
    replace_once(path, old, new)


def patch_train() -> None:
    path = ROOT / "vla_scripts/train.py"
    text = path.read_text()
    text = text.replace('tcad_negative_mode: str = "object_swap"', 'tcad_negative_mode: str = "manifest"')
    path.write_text(text)


def main() -> None:
    patch_oxe()
    patch_datasets()
    patch_data_utils()
    patch_base_strategy()
    patch_train()
    print("patched Spatial-LT OXE + dataset-aware TCAD")


if __name__ == "__main__":
    main()

from pathlib import Path

PATHS = [
    Path("prismatic/training/strategies/base_strategy.py"),
    Path("code/remote_files/base_strategy.py"),
]

OLD_SAMPLE = '''                    loss = output.loss
                    use_sample_weights = (
                        float(os.environ.get("RARE_BC_WEIGHT", "1.0")) != 1.0
                        or float(os.environ.get("TARGET_TASK_WEIGHT", "1.0")) != 1.0
                    )
                    if use_sample_weights and "sample_weights" in batch:
                        loss = self._weighted_action_loss(
                            output.logits,
                            batch["labels"],
                            batch["sample_weights"],
                            action_tokenizer,
                        )
                    tcad_loss = None
'''

NEW_SAMPLE = '''                    loss = output.loss
                    use_sample_weights = (
                        float(os.environ.get("RARE_BC_WEIGHT", "1.0")) != 1.0
                        or float(os.environ.get("TARGET_TASK_WEIGHT", "1.0")) != 1.0
                    )
                    confusion_gated_rare = os.environ.get("RARE_BC_CONFUSION_ONLY", "0").lower() in {
                        "1",
                        "true",
                        "yes",
                    }
                    effective_sample_weights = batch.get("sample_weights", None)
                    if use_sample_weights and not confusion_gated_rare and "sample_weights" in batch:
                        loss = self._weighted_action_loss(
                            output.logits,
                            batch["labels"],
                            batch["sample_weights"],
                            action_tokenizer,
                        )
                    tcad_loss = None
                    tcad_loss_term = None
                    corrective_active = None
'''

OLD_TCAD = '''                            if active.any():
                                margin = float(os.environ.get("TCAD_MARGIN", "0.2"))
                                tcad_loss = torch.relu(margin - (pos_score - neg_score))[active].mean()
                                loss = loss + tcad_weight * tcad_loss
                            else:
                                tcad_loss = torch.zeros((), device=output.logits.device)
                                loss = loss + 0.0 * neg_output.logits[:, :1, :1].sum()
                    anchor_l2_loss = self._anchor_l2_loss()
'''

NEW_TCAD = '''                            if active.any():
                                margin = float(os.environ.get("TCAD_MARGIN", "0.2"))
                                margin_loss = torch.relu(margin - (pos_score - neg_score))
                                corrective_active = active & (margin_loss.detach() > 0)
                                tcad_loss = margin_loss[active].mean()
                                tcad_loss_term = tcad_weight * tcad_loss
                                loss = loss + tcad_loss_term
                            else:
                                tcad_loss = torch.zeros((), device=output.logits.device)
                                loss = loss + 0.0 * neg_output.logits[:, :1, :1].sum()
                    if confusion_gated_rare and use_sample_weights and "sample_weights" in batch:
                        original_weights = batch["sample_weights"].to(output.logits.device).float()
                        gated_weights = torch.ones_like(original_weights)
                        if corrective_active is not None:
                            tail_mask = original_weights > 1.0
                            gated_weights = torch.where(corrective_active & tail_mask, original_weights, gated_weights)
                        effective_sample_weights = gated_weights
                        if bool((gated_weights != 1.0).any().item()):
                            loss = self._weighted_action_loss(
                                output.logits,
                                batch["labels"],
                                gated_weights,
                                action_tokenizer,
                            )
                            if tcad_loss_term is not None:
                                loss = loss + tcad_loss_term
                    anchor_l2_loss = self._anchor_l2_loss()
'''

OLD_DEBUG = '''                        sample_weights = batch.get("sample_weights", None)
                        if sample_weights is not None:
                            weights = sample_weights.float()
'''

NEW_DEBUG = '''                        sample_weights = effective_sample_weights
                        if sample_weights is not None:
                            weights = sample_weights.float()
'''


def patch_file(path: Path) -> bool:
    if not path.exists():
        print(f"skip missing {path}")
        return False
    text = path.read_text(encoding="utf-8")
    if "RARE_BC_CONFUSION_ONLY" in text:
        print(f"already patched {path}")
        return False
    for old, new, label in [
        (OLD_SAMPLE, NEW_SAMPLE, "sample-weight block"),
        (OLD_TCAD, NEW_TCAD, "tcad block"),
        (OLD_DEBUG, NEW_DEBUG, "debug block"),
    ]:
        if old not in text:
            raise SystemExit(f"anchor not found in {path}: {label}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")
    return True


if __name__ == "__main__":
    changed = False
    for path in PATHS:
        changed = patch_file(path) or changed
    print(f"changed={changed}")

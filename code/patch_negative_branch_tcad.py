from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "prismatic/training/strategies/base_strategy.py"
TRAIN = ROOT / "vla_scripts/train.py"

def patch_base(path):
    text = path.read_text(encoding="utf-8")
    if "TCAD_DETACH_POSITIVE" in text:
        print(f"already patched {path}")
        return
    text = text.replace(
'''                    tcad_loss = None
                    tcad_loss_term = None
                    corrective_active = None
                    tcad_active = batch.get("tcad_active", None)
''',
'''                    tcad_loss = None
                    tcad_loss_term = None
                    corrective_active = None
                    detach_positive_tcad = os.environ.get("TCAD_DETACH_POSITIVE", "0").lower() in {"1", "true", "yes"}
                    tcad_active = batch.get("tcad_active", None)
''', 1)
    text = text.replace(
'''                                margin = float(os.environ.get("TCAD_MARGIN", "0.2"))
                                margin_loss = torch.relu(margin - (pos_score - neg_score))
                                corrective_active = active & (margin_loss.detach() > 0)
''',
'''                                margin = float(os.environ.get("TCAD_MARGIN", "0.2"))
                                tcad_pos_score = pos_score.detach() if detach_positive_tcad else pos_score
                                margin_loss = torch.relu(margin - (tcad_pos_score - neg_score))
                                corrective_active = active & (margin_loss.detach() > 0)
''', 1)
    text = text.replace('''                                "weighted_count,mean_sample_weight,tcad_loss,anchor_l2_loss\\n"
''', '''                                "weighted_count,mean_sample_weight,tcad_loss,anchor_l2_loss,detach_positive\\n"
''', 1)
    text = text.replace('''                            f"{batch_size},{tail_hit_count},{weighted_count},{mean_sample_weight:.6f},"
                            f"{value},{anchor_value}\\n"
''', '''                            f"{batch_size},{tail_hit_count},{weighted_count},{mean_sample_weight:.6f},"
                            f"{value},{anchor_value},{int(detach_positive_tcad)}\\n"
''', 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")

def patch_train(path):
    text = path.read_text(encoding="utf-8")
    if "tcad_detach_positive" in text:
        print(f"already patched {path}")
        return
    text = text.replace(
'''    tcad_negative_mode: str = "manifest"                         # object_swap | relation_anchor
    rare_bc_max_count: int = 0                                      # If >0, upweight BC loss for tasks at/below this count
''',
'''    tcad_negative_mode: str = "manifest"                         # object_swap | relation_anchor
    tcad_detach_positive: bool = False                              # If true, TCAD gradients only suppress the negative-instruction branch
    rare_bc_max_count: int = 0                                      # If >0, upweight BC loss for tasks at/below this count
''', 1)
    text = text.replace(
'''    os.environ["TCAD_NEGATIVE_MODE"] = cfg.tcad_negative_mode
    os.environ["RARE_BC_MAX_COUNT"] = str(cfg.rare_bc_max_count if cfg.rare_bc_weight != 1.0 else 0)
''',
'''    os.environ["TCAD_NEGATIVE_MODE"] = cfg.tcad_negative_mode
    os.environ["TCAD_DETACH_POSITIVE"] = "1" if cfg.tcad_detach_positive else "0"
    os.environ["RARE_BC_MAX_COUNT"] = str(cfg.rare_bc_max_count if cfg.rare_bc_weight != 1.0 else 0)
''', 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")

if __name__ == "__main__":
    patch_base(BASE)
    patch_train(TRAIN)

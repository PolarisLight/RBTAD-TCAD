from pathlib import Path


ROOT = Path("/mnt/data/cyh/VLA-long-tail")

train_path = ROOT / "vla_scripts/train.py"
text = train_path.read_text()
text = text.replace(
    '    os.environ["TCAD_RATIO"] = "0.25" if cfg.tcad_lambda > 0 else "0.0"\n'
    '    os.environ["TCAD_LAMBDA"] = str(cfg.tcad_lambda)\n',
    '    os.environ["TCAD_RATIO"] = "1.0" if cfg.tcad_lambda > 0 else "0.0"\n'
    '    os.environ["TCAD_LAMBDA"] = str(cfg.tcad_lambda)\n',
    1,
)
if 'os.environ["TCAD_DEBUG_FILE"]' not in text:
    text = text.replace(
        '    if cfg.tcad_smoke_steps is not None:\n'
        '        os.environ["TCAD_SMOKE_STEPS"] = str(cfg.tcad_smoke_steps)\n'
        '    else:\n'
        '        os.environ.pop("TCAD_SMOKE_STEPS", None)\n',
        '    if cfg.tcad_smoke_steps is not None:\n'
        '        os.environ["TCAD_SMOKE_STEPS"] = str(cfg.tcad_smoke_steps)\n'
        '    else:\n'
        '        os.environ.pop("TCAD_SMOKE_STEPS", None)\n'
        '    os.environ["TCAD_DEBUG_FILE"] = str(run_dir / "tcad-debug.csv")\n',
        1,
    )
train_path.write_text(text)

strategy_path = ROOT / "prismatic/training/strategies/base_strategy.py"
text = strategy_path.read_text()
old = '''                    loss = output.loss
                    tcad_loss = None
                    tcad_active = batch.get("tcad_active", None)
                    if tcad_active is not None and tcad_active.any() and float(os.environ.get("TCAD_LAMBDA", "0")) > 0:
'''
new = '''                    loss = output.loss
                    tcad_loss = None
                    tcad_active = batch.get("tcad_active", None)
                    tcad_active_count = int(tcad_active.sum().item()) if tcad_active is not None else 0
                    if tcad_active is not None and tcad_active.any() and float(os.environ.get("TCAD_LAMBDA", "0")) > 0:
'''
if "tcad_active_count = int" not in text:
    text = text.replace(old, new, 1)

old = '''                if tcad_loss is not None:
                    metrics.commit(tcad_loss=tcad_loss.detach(), tcad_active_ratio=tcad_active.float().mean())
                loss.backward()
'''
new = '''                if tcad_loss is not None:
                    metrics.commit(tcad_loss=tcad_loss.detach(), tcad_active_ratio=tcad_active.float().mean())
                tcad_debug_file = os.environ.get("TCAD_DEBUG_FILE")
                if tcad_debug_file and overwatch.is_rank_zero():
                    with open(tcad_debug_file, "a") as f:
                        if metrics.global_step == 0:
                            f.write("step,active_count,batch_size,tcad_loss\\n")
                        value = "nan" if tcad_loss is None else f"{float(tcad_loss.detach().cpu()):.6f}"
                        batch_size = int(tcad_active.numel()) if tcad_active is not None else 0
                        f.write(f"{metrics.global_step},{tcad_active_count},{batch_size},{value}\\n")
                loss.backward()
'''
if "tcad_debug_file = os.environ.get" not in text:
    text = text.replace(old, new, 1)
strategy_path.write_text(text)

print("tcad debug patch applied")

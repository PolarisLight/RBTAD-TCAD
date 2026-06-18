from pathlib import Path


path = Path("/mnt/data/cyh/VLA-long-tail/prismatic/training/strategies/base_strategy.py")
text = path.read_text()
old = '''                if tcad_loss is not None:
                    metrics.commit(tcad_loss=tcad_loss.detach(), tcad_active_ratio=tcad_active.float().mean())
                tcad_debug_file = os.environ.get("TCAD_DEBUG_FILE")
'''
new = '''                tcad_debug_file = os.environ.get("TCAD_DEBUG_FILE")
'''
if old not in text:
    raise SystemExit("tcad metrics commit anchor not found")
text = text.replace(old, new, 1)
path.write_text(text)
print("tcad debug uses csv only; metrics commit removed")

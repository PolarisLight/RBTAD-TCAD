from pathlib import Path

path = Path("/mnt/data/cyh/VLA-long-tail/prismatic/models/load.py")
text = path.read_text()

needle = '    overwatch.info(f"Base vlm: {base_vlm}")\n'
patch = '''    if base_vlm == "pretrained/prism-qwen25-extra-dinosiglip-224px-0_5b":
        base_vlm = "prism-qwen25-extra-dinosiglip-224px+0_5b"

'''

if patch not in text:
    if needle not in text:
        raise SystemExit("needle not found")
    text = text.replace(needle, patch + needle, 1)
    path.write_text(text)

print("patched")

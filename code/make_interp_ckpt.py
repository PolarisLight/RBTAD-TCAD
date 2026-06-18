#!/usr/bin/env python3
import argparse
import os
import shutil

import torch


def should_include(path, include_prefixes):
    if not include_prefixes:
        return True
    joined = ".".join(path)
    return any(joined == prefix or joined.startswith(prefix + ".") for prefix in include_prefixes)


def could_contain_included(path, include_prefixes):
    if not include_prefixes:
        return True
    joined = ".".join(path)
    return any(prefix == joined or prefix.startswith(joined + ".") or joined.startswith(prefix + ".") for prefix in include_prefixes)


def interpolate_in_place(base_node, delta_node, alpha, include_prefixes=None, path=()):
    changed = 0
    skipped = 0
    if not isinstance(base_node, dict) or not isinstance(delta_node, dict):
        return changed, skipped

    for key, base_value in list(base_node.items()):
        if key not in delta_node:
            skipped += 1
            continue
        delta_value = delta_node[key]
        child_path = path + (str(key),)
        if not could_contain_included(child_path, include_prefixes):
            skipped += 1
            continue
        if isinstance(base_value, dict) and isinstance(delta_value, dict):
            sub_changed, sub_skipped = interpolate_in_place(base_value, delta_value, alpha, include_prefixes, child_path)
            changed += sub_changed
            skipped += sub_skipped
            continue
        if not should_include(child_path, include_prefixes):
            skipped += 1
            continue
        if not torch.is_tensor(base_value) or not torch.is_tensor(delta_value):
            skipped += 1
            continue
        if base_value.shape != delta_value.shape:
            skipped += 1
            continue
        if not torch.is_floating_point(base_value):
            base_node[key] = delta_value.clone()
            changed += 1
            continue
        mixed = base_value.float().mul(1.0 - alpha).add_(delta_value.float(), alpha=alpha)
        base_node[key] = mixed.to(dtype=base_value.dtype)
        changed += 1
    return changed, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--delta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--copy-config-from", required=True)
    parser.add_argument("--copy-stats-from", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--include-prefix", action="append", default=[])
    args = parser.parse_args()

    print(f"loading base={args.base}", flush=True)
    base_obj = torch.load(args.base, map_location="cpu")
    print(f"loading delta={args.delta}", flush=True)
    delta_obj = torch.load(args.delta, map_location="cpu")
    with torch.no_grad():
        changed, skipped = interpolate_in_place(base_obj, delta_obj, args.alpha, args.include_prefix)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.run_root, exist_ok=True)
    shutil.copy2(args.copy_config_from, os.path.join(args.run_root, "config.json"))
    shutil.copy2(args.copy_stats_from, os.path.join(args.run_root, "dataset_statistics.json"))
    torch.save(base_obj, args.out)
    print(f"saved={args.out}")
    print(f"alpha={args.alpha} include_prefix={args.include_prefix or ['<all>']} changed={changed} skipped={skipped}")


if __name__ == "__main__":
    main()

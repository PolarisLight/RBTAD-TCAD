#!/usr/bin/env python3
import sys
import torch

path = sys.argv[1]
obj = torch.load(path, map_location="cpu")
print(type(obj))
if isinstance(obj, dict):
    print("top_keys", list(obj.keys())[:30])
    for key, value in obj.items():
        if isinstance(value, dict):
            sample = list(value.keys())[:10]
            print("dict", key, "len", len(value), "sample", sample)
        elif torch.is_tensor(value):
            print("tensor", key, tuple(value.shape), value.dtype)
        else:
            print("item", key, type(value))

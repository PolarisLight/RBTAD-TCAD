import sys
import torch

ckpt = sys.argv[1]
obj = torch.load(ckpt, map_location="cpu")

def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (str(key),))
    elif torch.is_tensor(node):
        joined = ".".join(path)
        if "projector" in joined:
            print(joined, tuple(node.shape), node.dtype)

for _ in walk(obj):
    pass

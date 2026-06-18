from pathlib import Path


path = Path("/mnt/data/cyh/VLA-long-tail/prismatic/vla/datasets/datasets.py")
text = path.read_text()

old = """        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)
"""
new = """        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        negative_input_ids, negative_labels = torch.tensor(negative_input_ids), torch.tensor(negative_labels)
        pixel_values = self.image_transform(img)
"""
if old not in text:
    raise SystemExit("tensorize anchor not found")
text = text.replace(old, new, 1)

text = text.replace(
    """            negative_input_ids=torch.tensor(negative_input_ids),
            negative_labels=torch.tensor(negative_labels),
""",
    """            negative_input_ids=negative_input_ids,
            negative_labels=negative_labels,
""",
    1,
)

path.write_text(text)
print("fixed negative tensorize")

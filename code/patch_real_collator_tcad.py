from pathlib import Path


path = Path("/mnt/data/cyh/VLA-long-tail/prismatic/util/data_utils.py")
text = path.read_text()

old = '''        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
'''
new = '''        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        has_tcad = "negative_input_ids" in instances[0]
        if has_tcad:
            negative_input_ids = [instance["negative_input_ids"] for instance in instances]
            negative_labels = [instance["negative_labels"] for instance in instances]
            tcad_active = torch.stack([instance["tcad_active"] for instance in instances])
        if "dataset_name" in instances[0]:
'''
if "has_tcad = \"negative_input_ids\"" not in text:
    text = text.replace(old, new, 1)

old = '''        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)
'''
new = '''        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if has_tcad:
            negative_input_ids = pad_sequence(negative_input_ids, batch_first=True, padding_value=self.pad_token_id)
            negative_labels = pad_sequence(negative_labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        if has_tcad:
            negative_input_ids = negative_input_ids[:, : self.model_max_length]
            negative_labels = negative_labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)
        if has_tcad:
            negative_attention_mask = negative_input_ids.ne(self.pad_token_id)
'''
if "negative_attention_mask = negative_input_ids.ne" not in text:
    text = text.replace(old, new, 1)

old = '''        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
'''
new = '''        if has_tcad:
            output["negative_input_ids"] = negative_input_ids
            output["negative_attention_mask"] = negative_attention_mask
            output["negative_labels"] = negative_labels
            output["tcad_active"] = tcad_active
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
'''
if 'output["negative_input_ids"]' not in text:
    text = text.replace(old, new, 1)

path.write_text(text)
print("real collator tcad patch applied")

from pathlib import Path


path = Path("/mnt/data/cyh/VLA-long-tail/prismatic/util/data_utils.py")
text = path.read_text()

old = '''        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)
'''
new = '''        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
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
if old not in text:
    raise SystemExit("exact collator padding anchor not found")
text = text.replace(old, new, 1)
path.write_text(text)
print("force-fixed real collator negative padding")

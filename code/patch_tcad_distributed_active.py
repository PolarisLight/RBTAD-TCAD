from pathlib import Path


path = Path("/mnt/data/cyh/VLA-long-tail/prismatic/training/strategies/base_strategy.py")
text = path.read_text()

old = '''                    tcad_loss = None
                    tcad_active = batch.get("tcad_active", None)
                    tcad_active_count = int(tcad_active.sum().item()) if tcad_active is not None else 0
                    if tcad_active is not None and tcad_active.any() and float(os.environ.get("TCAD_LAMBDA", "0")) > 0:
                        active = tcad_active.to(output.logits.device)
                        neg_output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["negative_input_ids"],
                            attention_mask=batch["negative_attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["negative_labels"],
                        )
                        pos_score = self._tcad_action_logprob(output.logits, batch["labels"], action_tokenizer)
                        neg_score = self._tcad_action_logprob(
                            neg_output.logits, batch["negative_labels"], action_tokenizer
                        )
                        margin = float(os.environ.get("TCAD_MARGIN", "0.2"))
                        tcad_loss = torch.relu(margin - (pos_score - neg_score))[active].mean()
                        loss = loss + float(os.environ.get("TCAD_LAMBDA", "0")) * tcad_loss
'''
new = '''                    tcad_loss = None
                    tcad_active = batch.get("tcad_active", None)
                    tcad_active_count = int(tcad_active.sum().item()) if tcad_active is not None else 0
                    tcad_weight = float(os.environ.get("TCAD_LAMBDA", "0"))
                    if tcad_active is not None and tcad_weight > 0:
                        local_active = torch.tensor(
                            [1 if tcad_active.any() else 0],
                            device=output.logits.device,
                            dtype=torch.int32,
                        )
                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(local_active, op=dist.ReduceOp.MAX)
                        if int(local_active.item()) > 0:
                            active = tcad_active.to(output.logits.device)
                            neg_output: CausalLMOutputWithPast = self.vlm(
                                input_ids=batch["negative_input_ids"],
                                attention_mask=batch["negative_attention_mask"],
                                pixel_values=batch["pixel_values"],
                                labels=batch["negative_labels"],
                            )
                            pos_score = self._tcad_action_logprob(output.logits, batch["labels"], action_tokenizer)
                            neg_score = self._tcad_action_logprob(
                                neg_output.logits, batch["negative_labels"], action_tokenizer
                            )
                            if active.any():
                                margin = float(os.environ.get("TCAD_MARGIN", "0.2"))
                                tcad_loss = torch.relu(margin - (pos_score - neg_score))[active].mean()
                                loss = loss + tcad_weight * tcad_loss
                            else:
                                tcad_loss = torch.zeros((), device=output.logits.device)
                                loss = loss + 0.0 * neg_output.logits[:, :1, :1].sum()
'''

if old not in text:
    raise SystemExit("tcad distributed active anchor not found")
text = text.replace(old, new, 1)
path.write_text(text)
print("tcad distributed active patch applied")

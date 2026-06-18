from pathlib import Path


ROOT = Path("/mnt/data/cyh/VLA-long-tail")


def patch_datasets():
    path = ROOT / "prismatic/vla/datasets/datasets.py"
    text = path.read_text()

    for module in ("os", "random", "re"):
        if f"import {module}" not in text.splitlines()[:40]:
            text = text.replace("from dataclasses import dataclass\n", f"from dataclasses import dataclass\nimport {module}\n", 1)

    helper = r'''

TCAD_OBJECTS = [
    "black bowl", "cookie box", "plate", "ketchup", "basket", "alphabet soup",
    "stove", "cabinet", "cream cheese", "wine bottle", "rack",
]


def _tcad_make_wrong_instruction(instruction: str):
    instruction = instruction.lower()
    patterns = [
        r"^pick up the (.*?) next to the (.*?) and place it on the (.*?)$",
        r"^pick up the (.*?) on the (.*?) and place it on the (.*?)$",
        r"^pick up the (.*?) and place it in the (.*?)$",
        r"^push the (.*?) to the front of the (.*?)$",
        r"^put the (.*?) on top of the (.*?)$",
        r"^put the (.*?) on the (.*?)$",
        r"^put the (.*?) in the (.*?)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, instruction)
        if not match:
            continue
        groups = [g.strip() for g in match.groups()]
        target = groups[0]
        distractors = [g for g in groups[1:] if g and g != target]
        if distractors:
            return instruction.replace(target, distractors[0], 1)
    for obj in TCAD_OBJECTS:
        if obj in instruction:
            for replacement in TCAD_OBJECTS:
                if replacement != obj and replacement not in instruction:
                    return instruction.replace(obj, replacement, 1)
    return None
'''
    if "TCAD_OBJECTS" not in text:
        text = text.replace("IGNORE_INDEX = -100\n", "IGNORE_INDEX = -100\n" + helper, 1)

    old = '''        conversation = []

        # if there is no action horizon, remove it here.

        if self.action_tokenizer.required_future_horizon == 0:
            action = action[-1]
        else:
            # get the last FH + 1 actions (current action + future ones) if required
            action = action[-self.action_tokenizer.required_future_horizon - 1 :]

        tokenized_action = self.action_tokenizer(action)
        raw_action_tokens = self.base_tokenizer(tokenized_action)["input_ids"]

        conversation.extend(
            [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": tokenized_action},
            ]
        )
        num_answer_tokens = len(raw_action_tokens)

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        # print(prompt_builder.get_prompt())
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
'''
    new = '''        conversation = []

        # if there is no action horizon, remove it here.

        if self.action_tokenizer.required_future_horizon == 0:
            action = action[-1]
        else:
            # get the last FH + 1 actions (current action + future ones) if required
            action = action[-self.action_tokenizer.required_future_horizon - 1 :]

        tokenized_action = self.action_tokenizer(action)
        raw_action_tokens = self.base_tokenizer(tokenized_action)["input_ids"]

        conversation.extend(
            [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": tokenized_action},
            ]
        )
        num_answer_tokens = len(raw_action_tokens)

        # Construct Chat-based Prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        # print(prompt_builder.get_prompt())
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        tcad_ratio = float(os.environ.get("TCAD_RATIO", "0"))
        tcad_active = False
        negative_input_ids, negative_labels = input_ids, labels
        if tcad_ratio > 0 and random.random() < tcad_ratio:
            wrong_lang = _tcad_make_wrong_instruction(lang)
            # Smoke approximation of pre-contact: gripper is still open in the expert action.
            open_gripper = bool(np.asarray(action)[-1] < 0)
            if wrong_lang is not None and open_gripper:
                neg_conversation = [
                    {"from": "human", "value": f"What action should the robot take to {wrong_lang}?"},
                    {"from": "gpt", "value": tokenized_action},
                ]
                neg_prompt_builder = self.prompt_builder_fn("openvla")
                for turn in neg_conversation:
                    neg_prompt_builder.add_turn(turn["from"], turn["value"])
                negative_input_ids = self.base_tokenizer(
                    neg_prompt_builder.get_prompt(), add_special_tokens=True
                ).input_ids
                negative_labels = list(negative_input_ids)
                tcad_active = True
'''
    if "tcad_ratio = float(os.environ.get(\"TCAD_RATIO\"" not in text:
        text = text.replace(old, new, 1)

    old2 = '''        labels[: -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-num_end_tokens:] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)
'''
    new2 = '''        labels[: -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX
        negative_labels[: -(num_answer_tokens + num_end_tokens)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-num_end_tokens:] = IGNORE_INDEX
            negative_labels[-num_end_tokens:] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            negative_input_ids=torch.tensor(negative_input_ids),
            negative_labels=torch.tensor(negative_labels),
            tcad_active=torch.tensor(tcad_active, dtype=torch.bool),
            dataset_name=dataset_name,
        )
'''
    if "negative_input_ids=torch.tensor" not in text:
        text = text.replace(old2, new2, 1)

    old3 = '''        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
'''
    new3 = '''        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        has_tcad = "negative_input_ids" in instances[0]
        if has_tcad:
            negative_input_ids = [instance["negative_input_ids"] for instance in instances]
            negative_labels = [instance["negative_labels"] for instance in instances]
            tcad_active = torch.stack([instance["tcad_active"] for instance in instances])
'''
    if "has_tcad = \"negative_input_ids\"" not in text:
        text = text.replace(old3, new3, 1)

    old4 = '''        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)
'''
    new4 = '''        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
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
        text = text.replace(old4, new4, 1)

    old5 = '''        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
'''
    new5 = '''        if has_tcad:
            output["negative_input_ids"] = negative_input_ids
            output["negative_attention_mask"] = negative_attention_mask
            output["negative_labels"] = negative_labels
            output["tcad_active"] = tcad_active
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
'''
    if "output[\"negative_input_ids\"]" not in text:
        text = text.replace(old5, new5, 1)

    path.write_text(text)


def patch_train_config():
    path = ROOT / "vla_scripts/train.py"
    text = path.read_text()

    if "tcad_lambda: float" not in text:
        text = text.replace(
            "    seed: int = 7                                                   # Random seed (for reproducibility)\n",
            "    seed: int = 7                                                   # Random seed (for reproducibility)\n"
            "    tcad_lambda: float = 0.0                                        # TCAD ranking loss weight\n"
            "    tcad_margin: float = 0.2                                        # TCAD ranking margin\n"
            "    tcad_smoke_steps: Optional[int] = None                         # Stop early for smoke experiments\n",
            1,
        )

    env_block = '''    os.environ["TCAD_RATIO"] = "0.25" if cfg.tcad_lambda > 0 else "0.0"
    os.environ["TCAD_LAMBDA"] = str(cfg.tcad_lambda)
    os.environ["TCAD_MARGIN"] = str(cfg.tcad_margin)
    if cfg.tcad_smoke_steps is not None:
        os.environ["TCAD_SMOKE_STEPS"] = str(cfg.tcad_smoke_steps)
    else:
        os.environ.pop("TCAD_SMOKE_STEPS", None)

'''
    if 'os.environ["TCAD_RATIO"]' not in text:
        text = text.replace(
            "    # Get VLA Dataset & Collator\n",
            env_block + "    # Get VLA Dataset & Collator\n",
            1,
        )
    elif 'os.environ["TCAD_LAMBDA"]' not in text:
        text = text.replace(
            '    os.environ["TCAD_RATIO"] = str(max(0.0, cfg.tcad_lambda > 0 and 0.25 or 0.0))\n\n',
            env_block,
            1,
        )
    path.write_text(text)


def patch_strategy():
    path = ROOT / "prismatic/training/strategies/base_strategy.py"
    text = path.read_text()

    if "import os" not in text.splitlines()[:40]:
        text = text.replace("from abc import ABC, abstractmethod\n", "from abc import ABC, abstractmethod\nimport os\n", 1)

    if "def _tcad_action_logprob" not in text:
        marker = "    # === VLA Training ===\n"
        helper = r'''    def _tcad_action_logprob(self, logits, labels, action_tokenizer):
        action_logits = logits[:, self.vlm.vision_backbone.num_patches : -1].float()
        action_gt = labels[:, 1:].to(action_logits.device)
        mask = (action_tokenizer.action_token_end_idx > action_gt) & (
            action_gt > action_tokenizer.action_token_begin_idx
        )
        safe_gt = action_gt.clamp_min(0)
        log_probs = torch.log_softmax(action_logits, dim=-1)
        selected = log_probs.gather(-1, safe_gt.unsqueeze(-1)).squeeze(-1)
        selected = selected.masked_fill(~mask, 0.0)
        counts = mask.sum(dim=1).clamp_min(1)
        return selected.sum(dim=1) / counts

'''
        text = text.replace(marker, helper + marker, 1)

    old = '''                    loss = output.loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()
'''
    new = '''                    loss = output.loss
                    tcad_loss = None
                    tcad_active = batch.get("tcad_active", None)
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

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                if tcad_loss is not None:
                    metrics.commit(tcad_loss=tcad_loss.detach(), tcad_active_ratio=tcad_active.float().mean())
                loss.backward()
'''
    if "neg_output: CausalLMOutputWithPast" not in text:
        text = text.replace(old, new, 1)

    old2 = '''                if (terminate := (total_length is not None and metrics.global_step >= total_length +1)) or (
                    (metrics.global_step % save_interval) == 0
                ):
'''
    new2 = '''                smoke_steps = os.environ.get("TCAD_SMOKE_STEPS")
                smoke_terminate = smoke_steps is not None and metrics.global_step >= int(smoke_steps)
                if (terminate := ((total_length is not None and metrics.global_step >= total_length +1) or smoke_terminate)) or (
                    (metrics.global_step % save_interval) == 0
                ):
'''
    if "smoke_terminate = smoke_steps" not in text:
        text = text.replace(old2, new2, 1)

    path.write_text(text)


def main():
    patch_datasets()
    patch_train_config()
    patch_strategy()
    print("tcad smoke patches applied")


if __name__ == "__main__":
    main()

"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        save_every_n_steps: Optional[int] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None
        self._anchor_l2_params = None
        self._anchor_l2_numel = 0

        # how often to save checkpoints
        self.save_every_n_steps = save_every_n_steps
        if save_every_n_steps is not None:
            assert save_every_n_steps > 0

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return
                        elif (
                            self.save_every_n_steps is not None
                            and (metrics.global_step + 1) % self.save_every_n_steps == 0
                        ):

                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    def _tcad_action_logprob(self, logits, labels, action_tokenizer):
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

    def _weighted_action_loss(self, logits, labels, sample_weights, action_tokenizer):
        action_logits = logits[:, self.vlm.vision_backbone.num_patches : -1].float()
        action_gt = labels[:, 1:].to(action_logits.device)
        mask = (action_tokenizer.action_token_end_idx > action_gt) & (
            action_gt > action_tokenizer.action_token_begin_idx
        )
        safe_gt = action_gt.clamp_min(0)
        token_loss = torch.nn.functional.cross_entropy(
            action_logits.reshape(-1, action_logits.shape[-1]),
            safe_gt.reshape(-1),
            reduction="none",
        ).reshape_as(action_gt)
        token_loss = token_loss.masked_fill(~mask, 0.0)
        per_sample_loss = token_loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        weights = sample_weights.to(per_sample_loss.device).float()
        return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1.0)

    def init_anchor_l2_params(self):
        anchor_weight = float(os.environ.get("ANCHOR_L2_LAMBDA", "0"))
        if anchor_weight <= 0:
            self._anchor_l2_params = []
            return
        filters = [
            item.strip()
            for item in os.environ.get("ANCHOR_L2_FILTER", "").split(",")
            if item.strip() and item.strip().lower() not in {"none", "null", "<all>"}
        ]
        self._anchor_l2_params = []
        self._anchor_l2_numel = 0
        for name, param in self.vlm.named_parameters():
            if not param.requires_grad:
                continue
            if filters and not any(item in name for item in filters):
                continue
            anchor = param.detach().clone()
            self._anchor_l2_params.append((name, param, anchor))
            self._anchor_l2_numel += anchor.numel()
        if overwatch.is_rank_zero():
            overwatch.info(
                f"Anchor L2 enabled for {len(self._anchor_l2_params)} tensors "
                f"({self._anchor_l2_numel / 1e6:.3f}M local params), filters={filters or ['<all>']}"
            )

    def _anchor_l2_loss(self):
        anchor_weight = float(os.environ.get("ANCHOR_L2_LAMBDA", "0"))
        if anchor_weight <= 0:
            return None
        if self._anchor_l2_params is None:
            self.init_anchor_l2_params()
        if not self._anchor_l2_params or self._anchor_l2_numel <= 0:
            return None
        with torch.autocast("cuda", enabled=False):
            penalty = None
            updated = []
            for name, param, anchor in self._anchor_l2_params:
                if anchor.device != param.device:
                    anchor = anchor.to(param.device)
                value = (param.float() - anchor.float()).pow(2).sum()
                penalty = value if penalty is None else penalty + value
                updated.append((name, param, anchor))
            self._anchor_l2_params = updated
        return anchor_weight * penalty / max(self._anchor_l2_numel, 1)

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        world_size = self.global_batch_size//self.per_device_batch_size
        total_length = int(self.epochs * len(dataloader)//world_size)

        with tqdm(
            # total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            total=total_length,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss
                    use_sample_weights = (
                        float(os.environ.get("RARE_BC_WEIGHT", "1.0")) != 1.0
                        or float(os.environ.get("TARGET_TASK_WEIGHT", "1.0")) != 1.0
                    )
                    confusion_gated_rare = os.environ.get("RARE_BC_CONFUSION_ONLY", "0").lower() in {
                        "1",
                        "true",
                        "yes",
                    }
                    effective_sample_weights = batch.get("sample_weights", None)
                    if use_sample_weights and not confusion_gated_rare and "sample_weights" in batch:
                        loss = self._weighted_action_loss(
                            output.logits,
                            batch["labels"],
                            batch["sample_weights"],
                            action_tokenizer,
                        )
                    tcad_loss = None
                    tcad_loss_term = None
                    corrective_active = None
                    detach_positive_tcad = os.environ.get("TCAD_DETACH_POSITIVE", "0").lower() in {"1", "true", "yes"}
                    tcad_active = batch.get("tcad_active", None)
                    tcad_candidate_count = int(tcad_active.sum().item()) if tcad_active is not None else 0
                    tcad_active_count = tcad_candidate_count
                    tcad_weight = float(os.environ.get("TCAD_LAMBDA", "0"))
                    if tcad_active is not None and tcad_weight > 0:
                        active = tcad_active.to(output.logits.device)
                        pos_score = self._tcad_action_logprob(output.logits, batch["labels"], action_tokenizer)
                        if os.environ.get("TCAD_CONF_GATE", "none") == "batch_median" and active.any():
                            threshold = pos_score.detach().median()
                            active = active & (pos_score.detach() >= threshold)
                        tcad_active_count = int(active.sum().item())
                        local_active = torch.tensor(
                            [1 if active.any() else 0],
                            device=output.logits.device,
                            dtype=torch.int32,
                        )
                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(local_active, op=dist.ReduceOp.MAX)
                        if int(local_active.item()) > 0:
                            neg_output: CausalLMOutputWithPast = self.vlm(
                                input_ids=batch["negative_input_ids"],
                                attention_mask=batch["negative_attention_mask"],
                                pixel_values=batch["pixel_values"],
                                labels=batch["negative_labels"],
                            )
                            neg_score = self._tcad_action_logprob(
                                neg_output.logits, batch["negative_labels"], action_tokenizer
                            )
                            if active.any():
                                margin = float(os.environ.get("TCAD_MARGIN", "0.2"))
                                tcad_pos_score = pos_score.detach() if detach_positive_tcad else pos_score
                                margin_loss = torch.relu(margin - (tcad_pos_score - neg_score))
                                corrective_active = active & (margin_loss.detach() > 0)
                                tcad_loss = margin_loss[active].mean()
                                tcad_loss_term = tcad_weight * tcad_loss
                                loss = loss + tcad_loss_term
                            else:
                                tcad_loss = torch.zeros((), device=output.logits.device)
                                loss = loss + 0.0 * neg_output.logits[:, :1, :1].sum()
                    if confusion_gated_rare and use_sample_weights and "sample_weights" in batch:
                        original_weights = batch["sample_weights"].to(output.logits.device).float()
                        gated_weights = torch.ones_like(original_weights)
                        if corrective_active is not None:
                            tail_mask = original_weights > 1.0
                            gated_weights = torch.where(corrective_active & tail_mask, original_weights, gated_weights)
                        effective_sample_weights = gated_weights
                        if bool((gated_weights != 1.0).any().item()):
                            loss = self._weighted_action_loss(
                                output.logits,
                                batch["labels"],
                                gated_weights,
                                action_tokenizer,
                            )
                            if tcad_loss_term is not None:
                                loss = loss + tcad_loss_term
                    anchor_l2_loss = self._anchor_l2_loss()
                    if anchor_l2_loss is not None:
                        loss = loss + anchor_l2_loss

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                tcad_debug_file = os.environ.get("TCAD_DEBUG_FILE")
                if tcad_debug_file and overwatch.is_rank_zero():
                    with open(tcad_debug_file, "a") as f:
                        if metrics.global_step == 0:
                            f.write(
                                "step,candidate_count,active_count,batch_size,tail_hit_count,"
                                "weighted_count,mean_sample_weight,tcad_loss,anchor_l2_loss,detach_positive\n"
                            )
                        value = "nan" if tcad_loss is None else f"{float(tcad_loss.detach().cpu()):.6f}"
                        anchor_value = (
                            "nan"
                            if anchor_l2_loss is None
                            else f"{float(anchor_l2_loss.detach().cpu()):.6f}"
                        )
                        batch_size = int(tcad_active.numel()) if tcad_active is not None else 0
                        task_counts = batch.get("task_counts", None)
                        tail_limit = int(os.environ.get("TCAD_TAIL_MAX_COUNT", "0") or "0")
                        if task_counts is not None and tail_limit > 0:
                            tail_hit_count = int((task_counts.to(output.logits.device) <= tail_limit).sum().item())
                        else:
                            tail_hit_count = 0
                        sample_weights = effective_sample_weights
                        if sample_weights is not None:
                            weights = sample_weights.float()
                            weighted_count = int((weights != 1.0).sum().item())
                            mean_sample_weight = float(weights.mean().item())
                        else:
                            weighted_count = 0
                            mean_sample_weight = 1.0
                        f.write(
                            f"{metrics.global_step},{tcad_candidate_count},{tcad_active_count},"
                            f"{batch_size},{tail_hit_count},{weighted_count},{mean_sample_weight:.6f},"
                            f"{value},{anchor_value},{int(detach_positive_tcad)}\n"
                        )
                loss.backward()


                if True:

                    # === Compute Action Token Accuracy & L1 Loss ===

                    # To compute action token accuracy, we need to identify the locations of the action tokens
                    # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                    # insert `self.vlm.vision_backbone.num_patches` at index 1.
                    #
                    # Computing `action_prediction_accuracy` is then pretty straightforward:
                    #   1) Extract "aligned" predictions & labels
                    #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                    #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                    #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                    action_preds = output.logits[:, self.vlm.vision_backbone.num_patches : -1].argmax(dim=2)
                    action_gt = batch["labels"][:, 1:].to(action_preds.device)
                    mask = (action_tokenizer.action_token_end_idx > action_gt) & (action_gt > action_tokenizer.action_token_begin_idx)

                    # Compute Accuracy
                    correct_preds = (action_preds == action_gt) & mask
                    action_accuracy = correct_preds.sum().float() / mask.sum().float()

                    # Compute L1 Loss on Predicted (Continuous) Actions
                    actions_pred = action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                    if isinstance(actions_pred, list):
                        continuous_actions_pred = torch.cat(actions_pred)
                    else:
                        continuous_actions_pred = torch.tensor(actions_pred)

                    actions_gt = action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                    if isinstance(actions_gt, list):
                        continuous_actions_gt = torch.cat(actions_gt)
                    else:
                        continuous_actions_gt = torch.tensor(actions_gt)

                    action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                    # Commit Metrics
                    metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                    # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                    if overwatch.is_rank_zero():
                        datasets = set(batch["dataset_names"])
                        if len(datasets) > 1:
                            for ds in datasets:
                                ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                                action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()

                                actions_pred_ds = action_tokenizer.decode_token_ids_to_actions(
                                        action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                    )
                                if isinstance(actions_pred_ds, list):
                                    continuous_actions_pred_ds = torch.cat(actions_pred_ds)
                                else:
                                    continuous_actions_pred_ds = torch.tensor(actions_pred_ds)

                                actions_gt_ds = action_tokenizer.decode_token_ids_to_actions(
                                        action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                    )
                                if isinstance(actions_gt_ds, list):
                                    continuous_actions_gt_ds = torch.cat(actions_gt_ds)
                                else:
                                    continuous_actions_gt_ds = torch.tensor(actions_gt_ds)

                                action_l1_loss_ds = torch.nn.functional.l1_loss(
                                    continuous_actions_pred_ds, continuous_actions_gt_ds
                                )
                                metrics.commit_for_dataset(
                                    dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                                )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                limit_steps = os.environ.get("TRAIN_LIMIT_STEPS") or os.environ.get("TCAD_SMOKE_STEPS")
                limit_terminate = limit_steps is not None and metrics.global_step >= int(limit_steps)
                if (terminate := ((total_length is not None and metrics.global_step >= total_length +1) or limit_terminate)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)

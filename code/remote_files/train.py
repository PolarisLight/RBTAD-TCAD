"""
train.py

Training script for Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs, trained using mixtures of
the Open-X Embodiment dataset. Performs training in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed across GPUs (and nodes). By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`
    - If you want to suppress random Tensorflow logs --> `export TF_CPP_MIN_LOG_LEVEL=3`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/train.py
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['NO_GCE_CHECK'] = 'true'

import tensorflow as tf
tf.config.list_physical_devices('GPU')

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml

import sys
sys.path.append('.')
from prismatic.conf import VLAConfig
from prismatic.conf import VLARegistry
from prismatic.models import load, load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
# from datasets_with_vqa import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "/path/to/your/data_root"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Path] = None                    # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    #   (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    # max_steps: int = 50000
    max_steps: int = 10000000
    save_interval: int = 2500                                     # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 7                                                   # Random seed (for reproducibility)
    tcad_lambda: float = 0.0                                        # TCAD ranking loss weight
    tcad_ratio: float = 0.25                                        # TCAD candidate sampling ratio
    tcad_margin: float = 0.2                                        # TCAD ranking margin
    tcad_tail_max_count: int = 0                                    # If >0, enable TCAD only for tasks at/below this count
    tcad_conf_gate: str = "none"                                    # none | batch_median
    tcad_negative_mode: str = "manifest"                         # object_swap | relation_anchor
    tcad_detach_positive: bool = False                              # If true, TCAD gradients only suppress the negative-instruction branch
    rare_bc_max_count: int = 0                                      # If >0, upweight BC loss for tasks at/below this count
    rare_bc_weight: float = 1.0                                     # Per-sample BC weight for rare tasks
    tail_focal_lambda: float = 0.0                                   # Extra focal BC weight for low-confidence tail action tokens
    tail_focal_gamma: float = 2.0                                    # Focal exponent for tail_focal_lambda
    tail_focal_max_count: int = 0                                    # Enable tail focal BC for tasks at/below this count
    anchor_l2_lambda: float = 0.0                                   # L2-SP weight to keep short fine-tunes near the loaded checkpoint
    anchor_l2_filter: str = ""                                      # Comma-separated trainable-parameter name filters for anchor L2
    risk_bc_weight_manifest: str = ""                               # JSON mapping language instructions to closed-loop risk replay weights
    risk_bc_weights_json: str = ""                                  # Inline JSON alternative to risk_bc_weight_manifest
    bp_preserve_manifest: str = ""                                  # JSON manifest with bp_weight/tcad_enable per instruction
    bp_preserve_json: str = ""                                      # Inline JSON alternative to bp_preserve_manifest
    baseline_teacher_checkpoint: Optional[Path] = None                  # Frozen baseline teacher for behavior preservation
    bp_lambda: float = 0.0                                              # Action-token KL weight to preserve baseline behavior
    bp_temperature: float = 1.0                                         # KL distillation temperature
    bp_teacher_device: str = "cpu"                                      # cpu | cuda; CPU avoids replicated-teacher GPU OOM
    trainable_filter: str = ""                                      # Comma-separated parameter-name filters; non-matching params are frozen
    extra_trainable_filter: str = ""                                # Comma-separated frozen parameter-name filters to unfreeze after stage setup
    train_limit_steps: Optional[int] = None                         # Optional hard stop for short fine-tuning runs
    tcad_smoke_steps: Optional[int] = None                         # Stop early for smoke experiments

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", )                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "minivla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "username"                          # Name of entity to log under

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        self.epochs = self.vla.epochs
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy
        self.save_every_n_steps = self.vla.save_every_n_steps

        self.action_tokenizer = self.vla.action_tokenizer

        self.image_sequence_len = self.vla.image_sequence_len
        self.use_wrist_image = self.vla.use_wrist_image

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("OpenVLA Training :: Warming Up")

    # # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    # torch.cuda.set_device(device_id := overwatch.local_rank())
    # torch.cuda.set_device(0)
    # torch.cuda.empty_cache()
    
    is_distributed = int(os.environ.get("WORLD_SIZE", -1)) != -1
    if is_distributed:
        print("Running in DISTRIBUTED mode (torchrun)...")
        torch.cuda.set_device(device_id := overwatch.local_rank())
    else:
        print("Running in SINGLE-PROCESS mode...")
        device_id = 0
        torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        # if cfg.is_resume:
        #     assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_step
        #     assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint.name).group(1)) == cfg.resume_epoch

        vlm = load_vla(
            cfg.pretrained_checkpoint,
            hf_token=hf_token,
            load_for_training=True,
            image_sequence_len=cfg.image_sequence_len,
        )

    else:
        vlm = load(
            cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True, image_sequence_len=cfg.image_sequence_len
        )

    # [Validate] Model should be in Full Precision!
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-full-train"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "vla-train"  # Frozen vision encoder
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "You should unfreeze at least the last layer of your LLM!"
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        assert cfg.vla.unfreeze_last_llm_layer, "Need to unfreeze at least last LLM layer to train!"
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vlm.freeze_backbones(stage)

    extra_trainable_filters = [
        item.strip()
        for item in cfg.extra_trainable_filter.split(",")
        if item.strip() and item.strip().lower() not in {"none", "null"}
    ]
    if extra_trainable_filters:
        reopened_tensors = 0
        reopened_numel = 0
        for name, param in vlm.named_parameters():
            if any(item in name for item in extra_trainable_filters):
                if not param.requires_grad:
                    reopened_tensors += 1
                    reopened_numel += param.numel()
                param.requires_grad_(True)
        overwatch.info(
            f"Applied extra_trainable_filter={extra_trainable_filters}; reopened {reopened_tensors} tensors "
            f"({reopened_numel / 1e6:.3f}M params)"
        )

    trainable_filters = [
        item.strip()
        for item in cfg.trainable_filter.split(",")
        if item.strip() and item.strip().lower() not in {"none", "null", "<all>"}
    ]
    if trainable_filters:
        kept_tensors = 0
        kept_numel = 0
        frozen_tensors = 0
        for name, param in vlm.named_parameters():
            if param.requires_grad and any(item in name for item in trainable_filters):
                kept_tensors += 1
                kept_numel += param.numel()
            else:
                if param.requires_grad:
                    frozen_tensors += 1
                param.requires_grad_(False)
        overwatch.info(
            f"Applied trainable_filter={trainable_filters}; kept {kept_tensors} tensors "
            f"({kept_numel / 1e6:.3f}M params), additionally froze {frozen_tensors} tensors"
        )

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vlm.parameters())
    num_trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    os.environ["TCAD_RATIO"] = str(cfg.tcad_ratio if cfg.tcad_lambda > 0 else 0.0)
    os.environ["TCAD_LAMBDA"] = str(cfg.tcad_lambda)
    os.environ["TCAD_MARGIN"] = str(cfg.tcad_margin)
    os.environ["TCAD_TAIL_MAX_COUNT"] = str(cfg.tcad_tail_max_count if cfg.tcad_lambda > 0 else 0)
    os.environ["TCAD_CONF_GATE"] = cfg.tcad_conf_gate
    os.environ["TCAD_NEGATIVE_MODE"] = cfg.tcad_negative_mode
    os.environ["TCAD_DETACH_POSITIVE"] = "1" if cfg.tcad_detach_positive else "0"
    os.environ["RARE_BC_MAX_COUNT"] = str(cfg.rare_bc_max_count if cfg.rare_bc_weight != 1.0 else 0)
    os.environ["RARE_BC_WEIGHT"] = str(cfg.rare_bc_weight)
    os.environ["TAIL_FOCAL_LAMBDA"] = str(cfg.tail_focal_lambda)
    os.environ["TAIL_FOCAL_GAMMA"] = str(cfg.tail_focal_gamma)
    os.environ["TAIL_FOCAL_MAX_COUNT"] = str(cfg.tail_focal_max_count if cfg.tail_focal_lambda > 0 else 0)
    os.environ["ANCHOR_L2_LAMBDA"] = str(cfg.anchor_l2_lambda)
    os.environ["ANCHOR_L2_FILTER"] = cfg.anchor_l2_filter
    os.environ["RISK_BC_WEIGHT_MANIFEST"] = cfg.risk_bc_weight_manifest
    os.environ["RISK_BC_WEIGHTS_JSON"] = cfg.risk_bc_weights_json
    os.environ["BP_PRESERVE_MANIFEST"] = cfg.bp_preserve_manifest
    os.environ["BP_PRESERVE_JSON"] = cfg.bp_preserve_json
    os.environ["BP_LAMBDA"] = str(cfg.bp_lambda)
    os.environ["BP_TEMPERATURE"] = str(cfg.bp_temperature)
    if cfg.anchor_l2_lambda > 0 and cfg.pretrained_checkpoint is None:
        raise ValueError("anchor_l2_lambda > 0 requires pretrained_checkpoint as the anchor point")
    if cfg.train_limit_steps is not None:
        os.environ["TRAIN_LIMIT_STEPS"] = str(cfg.train_limit_steps)
    else:
        os.environ.pop("TRAIN_LIMIT_STEPS", None)
    if cfg.tcad_smoke_steps is not None:
        os.environ["TCAD_SMOKE_STEPS"] = str(cfg.tcad_smoke_steps)
    else:
        os.environ.pop("TCAD_SMOKE_STEPS", None)
    os.environ["TCAD_DEBUG_FILE"] = str(run_dir / "tcad-debug.csv")

    # Get VLA Dataset & Collator
    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vlm.vision_backbone.get_image_transform(),
        tokenizer=vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        action_tokenizer=cfg.action_tokenizer,
        # if using wrist images, we assume we passed in a 2x image sequence len
        image_window_size=cfg.image_sequence_len // 2 if cfg.use_wrist_image else cfg.image_sequence_len,
        use_wrist_image=cfg.use_wrist_image,  # will double the sequence length
    )
    print(f"Dataset length: {len(vla_dataset)}")

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
        save_every_n_steps=cfg.save_every_n_steps,
    )
    
    # import pdb; pdb.set_trace()
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))
    if cfg.anchor_l2_lambda > 0:
        train_strategy.init_anchor_l2_params()
    if cfg.bp_lambda > 0:
        if cfg.baseline_teacher_checkpoint is None:
            raise ValueError("bp_lambda > 0 requires baseline_teacher_checkpoint")
        overwatch.info(f"Loading frozen baseline teacher from `{cfg.baseline_teacher_checkpoint}`", ctx_level=1)
        baseline_teacher = load_vla(
            cfg.baseline_teacher_checkpoint,
            hf_token=hf_token,
            load_for_training=True,
            image_sequence_len=cfg.image_sequence_len,
        )
        for param in baseline_teacher.parameters():
            param.requires_grad_(False)
        baseline_teacher.eval()
        teacher_device = cfg.bp_teacher_device.strip().lower()
        if teacher_device == "cuda":
            baseline_teacher.to(torch.device("cuda", device_id))
        elif teacher_device != "cpu":
            raise ValueError(f"Unsupported bp_teacher_device={cfg.bp_teacher_device}; expected cpu or cuda")
        train_strategy.set_baseline_teacher(baseline_teacher, cfg.bp_lambda, cfg.bp_temperature)
    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        action_tokenizer,
        metrics,
        save_interval=cfg.save_interval,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":

    train()


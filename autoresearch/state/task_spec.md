# Spatial-LT Baseline vs RSDF Verification

Date: 2026-07-18

Goal: complete a theory-method-experiment-reflection loop for LIBERO-Spatial-LT and identify a simple method that improves over a matched BC baseline without adding inference-time modules or model complexity.

Remote execution:
- Host: `cyh@59.77.7.23:/mnt/data/cyh/VLA-long-tail`
- Environment: `/mnt/data/cyh/envs/vla-long-tail`
- GPUs: at most physical GPU 2 and 3; GPU 0/1 were not used by this run.

Success criteria:
- `libero_spatial_lt` resolves through OXE config/mixture/transform and TFDS.
- Baseline/RBTAD smoke runs validate dataloader, TCAD activation, and tail weighting.
- Matched baseline vs proposed method is evaluated under the same seed/checkpoint/eval protocol.
- The method improves the matched 30-trial baseline by at least 5 absolute points.
- All scripts, state, notes, and reproducible paths are persisted in the local repository.

Current best method:
- Relation-Localized Delta Fusion (RSDF), vision+LLM blend from baseline-1000 and BARC-p100, keeping projector fixed from baseline.

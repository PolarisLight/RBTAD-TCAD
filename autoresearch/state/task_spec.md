# Spatial-LT Baseline vs RBTAD Verification

Date: 2026-07-17

Goal: complete a runnable verification loop for LIBERO-Spatial-LT baseline vs RBTAD without occupying more than GPUs 2 and 3 on server 23.

Constraints:
- Remote: `cyh@59.77.7.23:/mnt/data/cyh/VLA-long-tail`
- Environment: `/mnt/data/cyh/envs/vla-long-tail`
- Use at most GPUs 2 and 3.
- Do not interrupt other users' processes.
- Do not treat single-seed screening as a final SOTA claim.

Minimum completion criteria:
- `libero_spatial_lt` is registered and dry-run dataset parsing passes.
- Baseline and RBTAD 5-step smoke both pass.
- RBTAD debug shows `TCAD active_count > 0`.
- Spatial-LT tail weighting is active.
- 1000-step matched screening is started or completed with reproducible paths.
- Code, state, and notes are committed locally and pushed unless blocked by a clear risk.

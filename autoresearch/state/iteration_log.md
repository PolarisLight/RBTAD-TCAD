# Iteration Log

## Iteration 1

Theory:
Long-tail VLA degradation should be tested as a conditional action-boundary problem, not only a frequency problem. On Spatial-LT, reusing Core-LT task counts or Core-specific object swaps would confound the diagnostic and can create invalid negatives.

Method:
Make dataset metadata explicit and dataset-aware. Valid counterfactual instructions must come from task instructions observed in the dataset or pass a semantic validity guard.

Experiment:
Pending dry-run and smoke.

Reflection:
Pending.
## Iteration 30 - CRGR closed-loop risk-gated replay

Theory:
RSDF has a real closed-loop relation-grounding signal, but the paired rollout diagnostic shows it can damage relation/action timing on other tasks. Red-team conclusion: do not use eval success to gate the method and do not add an inference-time module. Use calibration rollout behavior metrics only as a training-risk proxy, then test on heldout init ids.

Method:
CRGR starts from the protected RSDF checkpoint and performs a short end-to-end weighted BC recovery pass. The weights come from paired rollout behavior shifts: longer rollouts, reduced action norm, or strong gripper-close timing shift receive higher replay weight. Success labels are intentionally ignored in the weight generator. The policy architecture and inference path remain unchanged.

Experiment:
Server23 PID 3960780, GPUs 2/3 only. Risk manifest: /mnt/data/cyh/spatial_lt_crgr_risk_weights_20260719_192030.json. Both seed7 and seed13 5-step smoke runs passed; debug rows show weighted_count > 0 with mean_sample_weight around 1.4-1.7. Seed7 100-step run saved step-000100 checkpoint; seed13 100-step is running. Heldout eval uses fixed init ids 5..14, 10 trials/task, baseline vs RSDF vs CRGR.

Reflection:
This is a correction to the previous degrading-variant pattern. RSDF remains the protected current best until CRGR proves a cross-seed heldout gain. If CRGR fails, the failure should be interpreted as risk-weighted replay being too blunt, not as evidence for more parameter-fusion microvariants.

CRGR result update:
Seed7 heldout fixed-init result: baseline 0.16, RSDF 0.20, CRGR 0.21. Per-task CRGR: [0.50, 0.00, 0.30, 0.60, 0.20, 0.00, 0.50, 0.00, 0.00, 0.00].

Seed13 heldout fixed-init result: baseline 0.14, RSDF 0.13, CRGR 0.08. Per-task CRGR: [0.30, 0.00, 0.00, 0.30, 0.10, 0.00, 0.10, 0.00, 0.00, 0.00].

Reflection update:
CRGR is rejected as the final method. It confirms the theoretical critique: behavior-risk replay without an explicit baseline behavior-preservation term can overwrite relations that baseline still solves. The next method should preserve baseline actions on fragile/high-baseline relations while allowing RSDF-like correction only where rollout behavior suggests safe improvement.

## Iteration 31 - BPC-RSDF baseline-preserved correction

Theory:
CRGR failed because behavior-risk replay amplified biased closed-loop states and overwrote policies that RSDF or baseline still solved. The correction should therefore be constrained by an explicit non-regression signal instead of relying on initialization alone.

Method:
BPC-RSDF starts from the RSDF checkpoint and adds action-token KL from a frozen seed-matched baseline teacher only on manifest-protected instructions. Risk controls bp_weight/tcad_enable, not BC replay weight. TCAD remains detached-positive and low weight; sample_weight stays 1.0.

Experiment:
GPU teacher OOMed at smoke, so the active implementation moved the teacher to CPU. Seed7 and seed13 5-step smokes passed with bp_count>0, finite bp_loss, mean_sample_weight=1.0, and at least one TCAD active row. Server23 log: /mnt/data/cyh/spatial_lt_bpc_rsdf_screen_20260719_223327.log. The 50-step matched screen is running on GPUs 2/3 only.

Reflection:
This is slower but addresses the exact CRGR failure mode. Acceptance is non-regression against RSDF on both seeds; if either seed drops materially, reject it as another stabilizer/correction conflict rather than tuning weights blindly.
## Iteration 32 - BPC-RSDF 10-trial screen and 30-trial confirmation launch

Theory:
The CRGR failure predicted that replay-weighted correction damages seed-specific baseline competence. BPC-RSDF instead treats risk as a preservation gate: risky instructions are pulled toward a seed-matched baseline teacher while the student remains initialized from RSDF.

Method:
No inference-time module. A 50-step end-to-end correction from RSDF uses action-token KL to the frozen baseline teacher on manifest-protected instructions, low-weight detached-positive TCAD, and no BC replay upweight.

Experiment:
10-trial heldout init ids 5..14: seed7 baseline/RSDF/BPC = 0.16/0.20/0.24; seed13 = 0.14/0.13/0.19. Average gain over baseline is +6.5 points. Launched six sequential 30-trial confirmation evals: baseline, RSDF, BPC for seed7 and seed13. Log: /mnt/data/cyh/spatial_lt_bpc_rsdf_confirm30_20260721_014019.log.

Reflection:
This is the first candidate after RSDF that improves both seeds in the fixed heldout screen. It is not final proof yet: if 30-trial confirmation preserves the gain, BPC-RSDF becomes the current candidate method; otherwise classify it as a screening-only stabilizer.
## 2026-07-21 Iteration 31 confirm30 reflection

- Theory audit: BPC-RSDF assumed baseline action-token KL would prevent closed-loop behavior overwrite. Red-team failure: if RSDF's useful signal is itself a local action-distribution shift, teacher KL will over-preserve baseline errors and suppress relation correction.
- Method outcome: BPC-RSDF remains a diagnostic implementation branch only. It is not the new method.
- Experiment: seed7 30-trial baseline/RSDF/BPC = 0.21/0.25/0.19; seed13 30-trial baseline/RSDF = 0.13/0.15. BPC seed13 was stopped after seed7 rejection to release GPUs.
- Reflection: RSDF is still the protected current best (+3 points average over baseline) but fails the +5 target. The next cycle should target relation-specific rollout-state failures without global teacher preservation.

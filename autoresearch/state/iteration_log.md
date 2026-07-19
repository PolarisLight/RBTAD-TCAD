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

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

## Iteration 33 - GPRC gripper-phase relation contrast

Theory:
Red-team audit rejects another global preservation/replay or checkpoint-fusion variant. The remaining plausible failure mode is relation-conditioned phase timing: Spatial-LT keeps the manipulated object and goal nearly fixed, so source-relation errors must eventually surface in the gripper/close phase rather than in a broad action-token distribution. GPRC therefore avoids baseline-teacher KL, replay weights, and inference-time modules.

Method:
GPRC is standard BC plus a relation-neighbor TCAD margin computed only on the last valid action token (`tcad_token_scope=last_action`). Spatial-LT negatives are true in-dataset relation neighbors, not hash-random task instructions. The active screen starts from matched baseline-1000 checkpoints, uses `tcad_lambda=0.05`, `tcad_margin=0.10`, `tcad_tail_max_count=15`, `tcad_ratio=0.5`, and keeps `mean_sample_weight=1.0` / `bp_count=0`.

Experiment:
The first 20260721_075243 smoke was rejected because TCAD active_count stayed at zero. The fixed 20260721_075908 run uses ratio=1.0 only for 5-step smoke. Both smokes pass: seed7 has sum_candidate=6 and sum_active=6; seed13 has sum_candidate=7 and sum_active=7; both have `tcad_token_scope=last_action`, `mean_tcad_tokens=1.0`, `bp_count=0`, and `mean_sample_weight=1.0`. The 100-step matched screen is running on physical GPUs 2/3 only. Log: `/mnt/data/cyh/spatial_lt_gprc_screen_20260721_075908.log`.

Reflection:
The inactive first smoke prevented a false positive. GPRC remains a candidate, not a result. A 10-trial screen can only decide whether to launch 30-trial confirmation; final success still requires >=22.0% two-seed 30-trial average and no single-seed baseline regression.

## Iteration 34 - GPRC rejected; DP-GPRC next

Theory:
GPRC validates the gripper-phase relation signal on seed7 but fails the cross-seed stability test. The failure pattern is not global collapse: seed13 improves T4 from 0.00 baseline / 0.10 RSDF to 0.30, but damages T3 from 0.50/0.60 to 0.20 and T6 from 0.30/0.40 to 0.20. This suggests the direct positive-branch last-action margin over-corrects already-competent relation-phase behavior.

Method:
Reject direct GPRC as final. The next minimal candidate is DP-GPRC: keep true relation-neighbor negatives and last-action token scope, but set `tcad_detach_positive=True` so the margin mainly suppresses the wrong-instruction branch. It does not add replay weights, teacher KL, checkpoint fusion, or inference-time machinery.

Experiment:
GPRC 10-trial fixed-init screen completed on GPUs 2/3. Seed7 baseline/RSDF/GPRC = 0.16/0.20/0.26. Seed13 baseline/RSDF/GPRC = 0.14/0.13/0.09. Since seed13 regresses below the matched baseline, no 30-trial confirm is launched for direct GPRC.

Reflection:
The useful signal is phase-local, but the loss must avoid pushing the correct behavior branch directly. DP-GPRC is a structural response to the observed failure, not a broad hyperparameter sweep.

## 2026-07-28 Core-LT reviewer-priority reset

Theory:
The project state must stop treating Spatial-LT GPRC/DP-GPRC as the protected main path. The reviewer-priority closure is now Core-LT RBTAD-TCAD: same-pipeline APA reproduction, 2x2 ablation, multi-seed confirmation, APA+RBTAD complementarity, and margin-success diagnostics. Spatial-LT remains appendix/failure-boundary evidence.

Method:
P1/P2 execution is serialized on server23 physical GPU 2/3 only. Rare-BC seed7 completes the missing 2x2 ablation cell. APA is rebuilt from complete raw LIBERO Object/Goal files, not from partial no-noops or TFDS-only reconstructions.

Experiment:
Raw LIBERO-Object and LIBERO-Goal files are complete at 10/10 each. The old partial APA regeneration was stopped and moved aside. Robust APA queue is patched with `wait_for_rarebc_done` and is now gated before any GPU/no-noops/TFDS/training work. Rare-BC seed7 is alive on GPU 2/3.

Reflection:
The current evidence loop is not finished. Do not claim completion or update main-paper SOTA tables until same-pipeline APA and Rare-BC results are parsed. The protected main result remains seed7 Core-LT RBTAD about 40% versus local BC about 24%, with APA comparison pending.

## 2026-07-28 04:40 Core-LT Reviewer Priority Queue Persisted

- Rare-BC seed7 is alive on server23 physical GPU 2/3, latest observed step about 3029/34074.
- Robust full-raw APA remains gated behind Rare-BC; local reproducibility script persisted as `code/run_core_all_hf_download_then_apa_seed7_23.sh`.
- Post-APA priority queue persisted as `code/run_core_lt_priority_after_apa_multiseed_23.sh` and is waiting behind Rare-BC + APA.
- The post-APA queue order is matched Core-LT BC seed7, BC/RBTAD seed13, and BC/RBTAD seed21, all with 30 trials/task evaluation and physical GPU 2/3 only.

## 2026-07-28 04:45 Evidence Summarizer Added

- Added `code/summarize_core_lt_priority_results.py` to parse LIBERO `000.log` files into unified JSON/Markdown evidence summaries.
- Remote run generated:
  - `/mnt/data/cyh/VLA-long-tail/autoresearch/state/core_lt_priority_summary.json`
  - `/mnt/data/cyh/VLA-long-tail/autoresearch/state/core_lt_priority_summary.md`
- Local copies are stored at:
  - `autoresearch/state/core_lt_priority_summary.json`
  - `autoresearch/state/core_lt_priority_summary.md`
- Current parsed evidence:
  - legacy Core-LT BC seed7: 0.24;
  - TCAD-only/RCTAD seed7: 0.35;
  - RBTAD-TCAD seed7: 0.40;
  - matched BC seed7, Rare-BC seed7, APA seed7, seed13/21 BC/RBTAD remain pending because no matching `000.log` exists yet.

## 2026-07-28 04:59 Margin Diagnostic Queue Added

- Added `code/diagnose_core_lt_margins.py`, an offline Core-LT diagnostic that samples TCAD-eligible tail transitions, computes `log p(action|correct instruction) - log p(action|negative instruction)`, and correlates per-task margin deltas with per-task success deltas from `core_lt_priority_summary.json`.
- Added `code/run_core_lt_margin_diag_after_priority_23.sh`, a deferred server23 watcher that waits for Rare-BC, APA, and the post-APA multiseed queue to finish, then waits for physical GPU 2 to be idle and runs the diagnostic on a single allowed GPU.
- Remote watcher is alive and waiting:
  - process: 2903299;
  - script: `/mnt/data/cyh/run_core_lt_margin_diag_after_priority_23.sh`;
  - log: `/mnt/data/cyh/core_lt_margin_diag_after_priority_20260728.log`.
- The first launch exposed CRLF line ending fragility; local margin scripts were normalized to LF and `.gitattributes` now pins `*.sh text eol=lf`.

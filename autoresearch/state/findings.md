# Findings

Pending.
## 2026-07-19 CRGR launch finding

The degrading follow-up variants were caused by treating RSDF as a globally safe parameter delta. The paired rollout diagnostic instead indicates relation-specific closed-loop timing tradeoffs. CRGR tests a minimal end-to-end training correction: behavior-risk weighted replay from RSDF, no inference-time module, no success-label gating, and heldout fixed-init evaluation.

## 2026-07-19 CRGR negative result

CRGR improves seed7 heldout from 16% baseline / 20% RSDF to 21%, but collapses seed13 from 14% baseline / 13% RSDF to 8%. This rejects unprotected behavior-risk replay. The useful next problem is baseline behavior preservation under closed-loop correction, not additional replay weights or fusion microvariants.

## 2026-07-19 BPC-RSDF launch

BPC-RSDF is the next structural correction after CRGR rejection. It initializes from RSDF but replaces risk replay with baseline-preservation KL on manifest-protected instructions. The first GPU-teacher implementation OOMed under two-rank FSDP, so the active run uses a CPU frozen baseline teacher. Both seed7 and seed13 5-step smokes passed with bp_count>0, finite bp_loss, and mean_sample_weight=1.0. The 50-step matched screen is running on physical GPUs 2/3 only.
## 2026-07-21 BPC-RSDF positive screen

BPC-RSDF passes the 10-trial heldout non-regression gate. Seed7 improves from baseline/RSDF 0.16/0.20 to 0.24; seed13 improves from 0.14/0.13 to 0.19. Averaged over the two seeds, BPC-RSDF reaches 0.215 versus baseline 0.150 and RSDF 0.165. This is a +6.5 point screening gain over the matched baseline and +5.0 over RSDF. Because the goal requires reliable proof rather than a screening result, a six-way 30-trial confirmation has been launched.
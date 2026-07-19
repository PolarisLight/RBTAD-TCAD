# Findings

Pending.
## 2026-07-19 CRGR launch finding

The degrading follow-up variants were caused by treating RSDF as a globally safe parameter delta. The paired rollout diagnostic instead indicates relation-specific closed-loop timing tradeoffs. CRGR tests a minimal end-to-end training correction: behavior-risk weighted replay from RSDF, no inference-time module, no success-label gating, and heldout fixed-init evaluation.

## 2026-07-19 CRGR negative result

CRGR improves seed7 heldout from 16% baseline / 20% RSDF to 21%, but collapses seed13 from 14% baseline / 13% RSDF to 8%. This rejects unprotected behavior-risk replay. The useful next problem is baseline behavior preservation under closed-loop correction, not additional replay weights or fusion microvariants.

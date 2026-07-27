# Core-LT Priority Evidence Summary

## Seed7 Main / 2x2
| Cell | Overall | Per-task | Log |
| --- | ---: | --- | --- |
| BC seed7 legacy local | 0.24 | 0.63, 0.37, 0.23, 0.40, 0.43, 0.23, 0.00, 0.07, 0.03, 0.00 | `/mnt/data/cyh/VLA-long-tail/results/miniVLA_libero_core_lt/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-core-lt+n0+b10+x7/17038/baseline_lt_step17038_30trials_egl_20260607_132054/libero_core-prismatic/step_17038-vqa_False/000.log` |
| BC seed7 matched two-GPU | pending | pending | `pending` |
| TCAD-only seed7 | 0.35 | 0.47, 0.70, 0.70, 0.60, 0.33, 0.33, 0.07, 0.00, 0.27, 0.03 | `/mnt/data/cyh/VLA-long-tail/results/rctad_lt_main/rctad_tail9_confmedian_ratio05_seed7_b20/034075/rctad_step034075_30trials_egl_20260609_052111/libero_core-prismatic/step_34075-vqa_False/000.log` |
| Rare-BC seed7 | pending | pending | `pending` |
| RBTAD-TCAD seed7 | 0.40 | 0.60, 0.53, 0.43, 0.57, 0.50, 0.30, 0.60, 0.13, 0.37, 0.00 | `/mnt/data/cyh/VLA-long-tail/results/rbtad_lt_main/rbtad_w3_tail9_confmedian_seed7_b20/034075/rbtad_step034075_30trials_egl_20260610_014339/libero_core-prismatic/step_34075-vqa_False/000.log` |
| APA seed7 same pipeline | pending | pending | `pending` |

Seed7 RBTAD gain vs legacy BC: 0.16
Seed7 RBTAD gain vs matched BC: pending

## Multiseed BC vs RBTAD
| Seed | BC | RBTAD | Delta | Status |
| ---: | ---: | ---: | ---: | --- |
| 7 | 0.24 | 0.40 | 0.16 | done |
| 13 | pending | pending | pending | pending |
| 21 | pending | pending | pending | pending |
| avg n=1 | 0.24 | 0.40 | 0.16 | partial |

## Notes
- `pending` means no matching `000.log` has been produced yet.
- The strict 2x2 table should prefer matched two-GPU BC when available; legacy BC is retained to protect continuity with the current draft result.

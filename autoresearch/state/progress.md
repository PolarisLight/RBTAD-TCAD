# Progress

- 2026-07-17: Initialized state tracking.
- 2026-07-19 19:31: Iteration 30 launched CRGR (Closed-loop Risk-Gated Replay) on server23, PID 3960780, log /mnt/data/cyh/spatial_lt_crgr_screen_20260719_192030.log. Both 5-step smoke runs passed with risk-weighted samples active. Seed7 100-step checkpoint has been saved; seed13 100-step is running. GPUs are restricted to physical 2/3.
- 2026-07-19 22:12: CRGR heldout screen completed and was rejected as final. Seed7: baseline 0.16, RSDF 0.20, CRGR 0.21. Seed13: baseline 0.14, RSDF 0.13, CRGR 0.08. GPUs released. Next direction is baseline-preserving closed-loop correction, not more unprotected replay.

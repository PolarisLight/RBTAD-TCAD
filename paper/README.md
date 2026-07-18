# RBTAD / TCAD arXiv Draft

This directory contains the first arXiv-style manuscript draft for the long-tailed embodied imitation learning project.

Main file:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

Compiled draft PDF:

```text
rbtad_tcad_arxiv_draft_v2.pdf
```

Current claim boundary:

- RBTAD / TCAD is the main end-to-end method.
- RSDF vision+LLM is included as the current Spatial-LT screening method; it is a simple checkpoint-delta fusion without inference-time modules.
- LIBERO-Core-LT, LIBERO-Core-Full, and matched LIBERO-Spatial-LT screening numbers are included.
- Multi-seed and exact-protocol checks are listed as required before a strong SOTA claim.

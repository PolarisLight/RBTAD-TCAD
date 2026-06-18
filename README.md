# RBTAD-TCAD

This repository contains the working code and manuscript sources for **Rare-Balanced Task-Conditioned Action Discrimination (RBTAD)**, a preliminary training-only method for long-tailed embodied imitation learning.

The current manuscript is a technical-report draft rather than a finalized benchmark paper. Its main empirical evidence is based on LIBERO-Core-LT and a controlled LIBERO-Core-Full counterpart experiment; multi-seed evaluation and additional simulated long-tail splits are still required before making a strong SOTA claim.

## Contents

- `code/`: training, evaluation, patching, diagnostic, and remote-launch scripts used during the reproduction and method exploration cycle.
- `paper/`: LaTeX manuscript source, references, figure scripts, source data, and the current compiled draft PDF.
- `paper/rbtad_tcad_arxiv_draft_v2.pdf`: current compiled technical-report draft.

## Build the Manuscript

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

## Claim Boundary

RBTAD/TCAD is the main end-to-end method in this draft. Selective projector merge and other variants are retained only as diagnostic evidence, not as the proposed method. The repository intentionally excludes local transfer archives, generated environments, LaTeX build products, and large raster exports.

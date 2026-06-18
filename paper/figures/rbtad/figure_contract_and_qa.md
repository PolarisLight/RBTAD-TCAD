# RBTAD Figure Contract and QA Notes

## Figure Contract

Core conclusion:
RBTAD improves long-tail VLA fine-tuning by explicitly training a task-conditioned action boundary, not by adding generated demonstrations or inference-time modules.

Figure archetype:
Quantitative grid for result figures; schematic-led composite for the method framework.

Target journal/output:
arXiv draft with Nature/NeurIPS-compatible editable scientific figures. Quantitative figures are exported as SVG, PDF, PNG, and TIFF. The method framework is a raster imagegen draft and should be redrawn as vector before final camera-ready submission.

Backend:
Python/matplotlib for quantitative figures. Built-in imagegen for the method framework raster draft.

Final size:
Double-column-width landscape figures around 7.1 inches wide for the main plots; compact supplementary figure around 5.2 inches wide.

Panel map:
- Fig. 1a: LIBERO-Core-LT long-tail demonstration distribution.
- Fig. 1b: Instruction-swap diagnostic showing non-positive versus positive margins.
- Fig. 3 draft: Behavior cloning, counterfactual instruction pair, and margin ranking loss.
- Fig. 4a: Main LIBERO-Core-LT comparison against the reported baseline family.
- Fig. 5a: Core-Full within-pipeline overall comparison.
- Fig. 5b: Core-Full per-task paired comparison.
- Fig. S: Internal iteration summary.

Evidence hierarchy:
- Hero evidence: Fig. 4, because it places RBTAD against the reported APA baseline family.
- Mechanism evidence: Fig. 1 diagnostic and Fig. 3 method framework.
- Control evidence: Fig. 5 Core-Full controlled comparison.
- Design-path evidence: Fig. S internal iterations.

Statistics needed:
Current results are single-seed success rates. No error bars are plotted because no multi-seed variance is available yet. The figure captions should state rollout counts and single-seed status.

Source data needed:
Stored in `source_data.csv`.

Image-integrity notes:
No external task screenshots are used in this first figure set. The method framework is generated raster artwork and should not be treated as raw experimental evidence.

Reviewer risk:
Single-seed results remain the main risk. The method framework should be redrawn as editable vector before final submission. Simulation task thumbnails still need to be extracted from our own rollouts rather than copied from APA.

## QA Notes

- Avoided dual y axes, pie charts, 3D plots, and rainbow colormaps.
- Success-rate axes start at zero.
- Per-task Core-Full comparison uses paired dots rather than a categorical trend line.
- Quantitative exports include SVG/PDF for editable vector text.
- The method framework text is readable in the generated raster draft, but not editable.

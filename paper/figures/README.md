# Figures

wsm_method_overview.svg is an authored vector SVG and wsm_method_overview.pdf is its local vector PDF export for direct pdflatex inclusion. Neither file embeds raster artwork.

The revised figure is a compact two-panel conference-paper-style diagram informed only by the general visual readability of the supplied docs/Ryumin_EMNLP.pdf manuscript (clean grouped modules, restrained accents, and clear inference versus training paths). It does not copy that manuscript's figures, content, or layout.

Panel (a) depicts the primary equal-parameter shared-fusion inference path: WavLM audio and DEPART-like CLIP video features, projections, the averaged disease query, shared candidates, an availability-aware gate, fused representation, and independent depression/Parkinson heads. A small inset distinguishes full R4's task-specific query/gating from the primary shared version.

Panel (b) depicts frozen training supervision only: observed masked labels, the immutable accepted-pseudo cache, detached reliability, audio/video auxiliary heads, agreement, per-task objectives, and RA-STCH. Dashed arrows denote training-only paths. The figure explicitly notes that RA-STCH is retained in the frozen recipe but its contribution is not supported by ablation; it does not portray semantic evidence as essential.

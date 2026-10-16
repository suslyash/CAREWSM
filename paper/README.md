# WSM Paper Draft

## Status

**Internal venue-neutral submission-ready draft.**

Title: *Reliability-Weighted Cross-Corpus Pseudo-Supervision for Multimodal
Depression and Parkinson Recognition under Partial Labels*.

This is not an official ACL, EMNLP, or other venue-formatted submission. A
target venue has not been selected.

## Authoritative paper files

- Source entrypoint: `main.tex`
- Compiled release artifact: `release/WSM_paper_draft.pdf`
- Vector method figure: `figures/wsm_method_overview.svg` and
  `figures/wsm_method_overview.pdf`
- Reproducibility appendix: `sections/appendix_reproducibility.tex`
- Release manifest: `release/SOURCE_MANIFEST.md`
- Submission checklist: `SUBMISSION_CHECKLIST.md`

## Build

From this directory, run the required sequence:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The expected output is `main.pdf` with 14 pages. The tracked release PDF is
the clean build at `release/WSM_paper_draft.pdf`.

## Frozen scientific roles

- Equal-parameter shared fusion is the primary parsimonious paper candidate,
  selected before Final Test by a DEV-only parsimony decision.
- Full R4 trial012 is the secondary multimodal reference.
- Frozen temporal audio is the baseline.
- T1 is standalone text evidence only; T2 is blocked before cache/model/training.

All predeclared paired Final-Test Mean 95% confidence intervals include zero;
the manuscript makes no Final-Test statistical-superiority claim.

## Template-conversion boundary

Official venue formatting has **not** been applied. A later template conversion
must not change frozen scientific results, claims, roles, thresholds, or
evidence boundaries. See `SUBMISSION_CHECKLIST.md` for venue-dependent work.

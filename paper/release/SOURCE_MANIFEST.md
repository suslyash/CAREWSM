# WSM Paper Release Source Manifest

## Release artifact

- PDF: `WSM_paper_draft.pdf`
- SHA256: `fca4fcc90e4107592d8b1ce0f46bd8825d67476b58074fa5575f0cb2ee74eefd`
- Pages: `14`
- Build: `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` from `paper/`

## Source identity

- Release source base (`origin/main`):
  `dc2d348e1ce50620891e2a5ced7d38bd99c0cd2b`
- TASK-008C final commit: recorded in the Git handoff after release commit.

## Required rebuild files

- `main.tex`
- `sections/abstract.tex`
- `sections/introduction.tex`
- `sections/related_work.tex`
- `sections/data_and_protocol.tex`
- `sections/method.tex`
- `sections/experiments.tex`
- `sections/results.tex`
- `sections/limitations.tex`
- `sections/conclusion.tex`
- `sections/appendix_reproducibility.tex`
- `references.bib`
- `figures/wsm_method_overview.pdf`
- `figures/wsm_method_overview.svg` (authoritative vector source for the PDF)

## Asset checksums

| Asset | SHA256 |
|---|---|
| `WSM_paper_draft.pdf` | `fca4fcc90e4107592d8b1ce0f46bd8825d67476b58074fa5575f0cb2ee74eefd` |
| `figures/wsm_method_overview.pdf` | `85b543f7835fc8b9943b040a11a0e8f416f2e2bf22ee4ec7e7d77fca7d1598b0` |
| `figures/wsm_method_overview.svg` | `ec9fc0f81ddc78c57450c80de82f1d7cbf368d2a52f90acef1078265ea6010d5` |
| `references.bib` | `46b311746f92bb7562cafc1ec0b5bc2e8ba2bafb022dd293f5a39a1ab8a30391` |

No unrelated repository files are required to rebuild the manuscript.

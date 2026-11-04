# REFACTOR-004 — Mandatory Logging Simplification

## Scope

REFACTOR-004 removes the custom logging suppression introduced by REFACTOR-003. `CAREWSMSummaryCallback` now uses the original Chimera Trainer logger directly. It does not replace `trainer.logger`, install handler filters, match log prefixes, or emit duplicate corpus INFO records. Native Chimera Trainer execution owns the five epoch records for `train`, `val_care`, `val_wsm`, `test_care`, and `test_wsm`.

The callback still evaluates fixed TEST populations through Chimera's existing `_run_epoch`, collects the returned metrics, writes one wide semicolon-delimited row per epoch, and preserves the previously approved idempotency and legacy summary compatibility. Corpus loader names are passed through unchanged; legacy `CARE`/`WSM` names use explicit compatibility mappings.

## Verification

- `.venv/bin/pytest -q tests/test_refactor_summary.py` — 4 passed.
- The focused tests verify native five-record ordering, no callback-generated INFO records, one TEST evaluation per corpus per epoch, one summary row, unchanged Trainer logger state on TEST failure, and no installed filters.
- `git diff --check` — passed.

No real CARE/WSM training was run and no historical experiment output was changed. Remaining full-suite and integration checks are reported in the REFACTOR-004 handoff.

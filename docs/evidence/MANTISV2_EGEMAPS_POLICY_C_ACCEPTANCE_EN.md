# MantisV2 Policy C eGeMAPS Temporal Acceptance

Date: 2026-11-06
Required main: `d5758eaa824da2daad0198f97b11d906bc1c8cfb`
Branch: `codex/mantisv2-egemaps-policy-c-validator`

## 1. Decision

**GO for the eGeMAPS temporal-data contract under the explicit Policy C validator mode only.**

This GO authorizes only a separately reviewed MantisV2 pilot using in-memory terminal-frame exclusion. It does not establish MantisV2 package compatibility, checkpoint compatibility, model performance, or permission to install MantisV2. Strict mode remains the default and preserves the documented NO-GO result.

Policy C passed the canonical CARE/WSM TRAIN/VAL audit with no residual integrity failures. It excluded only the previously characterized CARE terminal boundary frames and did not modify source CSVs, timestamps, splits, or historical results.

## 2. Policy C contract

The validator now supports `--boundary-policy strict` (default) and `--boundary-policy terminal-exclude`.

A frame is excluded only when all conditions hold:

- corpus is CARE;
- it is the sole final frame crossing its turn boundary;
- its frame start is inside the declared turn;
- its frame end exceeds the turn end beyond the numerical tolerance;
- excess is no more than `1 / 16000 + 1e-6 = 0.0000635` seconds;
- turn metadata is consistent;
- original cadence, duplicate, ordering, feature, timestamp, schema, and profile checks pass;
- no other boundary violation exists in that turn or artifact.

The reason is recorded as `excluded_terminal_out_of_turn_1sample`. The original frame sequence is validated before masking. Excluded frames do not contribute to retained interval coverage or expected-frame density. No clipping, timestamp alteration, interpolation, or source-file rewrite occurs.

The future cache contract now includes explicit `frame_exclusion_mask` and `frame_exclusion_reason` fields alongside embeddings, window times, validity masks, and valid-observation fractions.

## 3. Synthetic acceptance

The real artifact inspection logic was exercised with temporary synthetic CSV fixtures. Passed cases include:

- valid 10 ms start cadence and 20 ms ordinary frames;
- permitted one-sample terminal excess;
- excess above the approved limit;
- nonterminal boundary crossing;
- multiple boundary violations;
- duplicate and missing starts;
- inconsistent turn metadata;
- nonfinite feature values;
- incorrect channel order;
- short turns and boundary-aligned windows;
- strict rejection versus Policy C exclusion;
- explicit exclusion reason, retained-row count, post-mask density, and cache schema.

Strict mode rejects the permitted terminal excess. Policy C excludes exactly one permitted frame. Both modes reject unrelated anomalies.

## 4. Real audit results

Only canonical CARE and WSM `fold_{0..4}_{train,val}.csv` populations and manifest-valid eGeMAPS artifacts were read. TEST data were not accessed.

Strict mode reproduced the prior behavior:

- CARE invalid artifact occurrences: 155;
- CARE original and residual out-of-turn rows: 165 / 165;
- Policy C exclusions: 0;
- strict CARE windows: 1,830,705 candidate; 1,830,680 accepted at 80%; 1,830,575 accepted at 95%;
- strict WSM windows: 4,089,690 candidate and accepted at both thresholds;
- no missing expected starts, duplicate timestamps, nonfinite features, inconsistent turn boundaries, or profile overlap.

Policy C completed with an empty failure list:

- unique excluded CARE artifacts: 31;
- unique excluded CARE turns: 33;
- excluded frame occurrences across the ten fold populations: 165;
- excluded reason: 165 `excluded_terminal_out_of_turn_1sample`;
- original out-of-turn rows: 165;
- residual out-of-turn rows after masking: 0;
- unrelated invalid rows/events: 0;
- CARE fully valid artifact occurrences: all manifest-valid occurrences;
- WSM excluded frames and out-of-turn rows: 0.

## 5. Policy C fold and split coverage

| Corpus | Fold | Split | Source | Manifest | Full | Excluded | Candidate / accepted 80% / accepted 95% | Density before → after (mean) | Min after |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| CARE | 0 | TRAIN | 1361 | 1361 | 1361 | 23 | 298440 / 298436 / 298419 | 0.999948264 → 0.999948231 | 0.60 |
| CARE | 0 | VAL | 352 | 352 | 352 | 10 | 74631 / 74630 / 74625 | 0.999935282 → 0.999935148 | 0.73 |
| CARE | 1 | TRAIN | 1368 | 1368 | 1368 | 28 | 292532 / 292527 / 292506 | 0.999940177 → 0.999940109 | 0.60 |
| CARE | 1 | VAL | 345 | 345 | 345 | 5 | 80539 / 80539 / 80538 | 0.999965607 → 0.999965607 | 0.76 |
| CARE | 2 | TRAIN | 1378 | 1378 | 1378 | 28 | 300814 / 300809 / 300791 | 0.999943819 → 0.999943753 | 0.60 |
| CARE | 2 | VAL | 335 | 335 | 335 | 5 | 72257 / 72257 / 72253 | 0.999953361 → 0.999953361 | 0.77 |
| CARE | 3 | TRAIN | 1382 | 1382 | 1382 | 29 | 300605 / 300601 / 300584 | 0.999945277 → 0.999945210 | 0.67 |
| CARE | 3 | VAL | 331 | 331 | 331 | 4 | 72466 / 72465 / 72460 | 0.999947286 → 0.999947286 | 0.60 |
| CARE | 4 | TRAIN | 1363 | 1363 | 1363 | 24 | 299893 / 299891 / 299876 | 0.999950682 → 0.999950649 | 0.60 |
| CARE | 4 | VAL | 350 | 350 | 350 | 9 | 73178 / 73175 / 73168 | 0.999925114 → 0.999924977 | 0.67 |
| WSM | 0 | TRAIN | 5640 | 5637 | 5637 | 0 | 652246 / 652246 / 652246 | 0.999984822 → 0.999984822 | 0.96 |
| WSM | 0 | VAL | 1429 | 1428 | 1428 | 0 | 165692 / 165692 / 165692 | 0.999985153 → 0.999985153 | 0.96 |
| WSM | 1 | TRAIN | 5661 | 5658 | 5658 | 0 | 654500 / 654500 / 654500 | 0.999985073 → 0.999985073 | 0.96 |
| WSM | 1 | VAL | 1408 | 1407 | 1407 | 0 | 163438 / 163438 / 163438 | 0.999984153 → 0.999984153 | 0.96 |
| WSM | 2 | TRAIN | 5670 | 5666 | 5666 | 0 | 656033 / 656033 / 656033 | 0.999985336 → 0.999985336 | 0.96 |
| WSM | 2 | VAL | 1399 | 1399 | 1399 | 0 | 161905 / 161905 / 161905 | 0.999983076 → 0.999983076 | 0.96 |
| WSM | 3 | TRAIN | 5640 | 5637 | 5637 | 0 | 652950 / 652950 / 652950 | 0.999984348 → 0.999984348 | 0.96 |
| WSM | 3 | VAL | 1429 | 1428 | 1428 | 0 | 164988 / 164988 / 164988 | 0.999987029 → 0.999987029 | 0.96 |
| WSM | 4 | TRAIN | 5665 | 5662 | 5662 | 0 | 656023 / 656023 / 656023 | 0.999984863 → 0.999984863 | 0.96 |
| WSM | 4 | VAL | 1404 | 1403 | 1403 | 0 | 161915 / 161915 / 161915 | 0.999984992 → 0.999984992 | 0.96 |

Across CARE fold populations, Policy C produced 1,865,355 candidate windows, 1,865,330 accepted at 80%, and 1,865,220 accepted at 95%. Across WSM fold populations, 4,089,690 windows were accepted at both thresholds. CARE post-mask density minima ranged from 0.60 to 0.77; WSM minima were 0.96. The small density decreases in affected CARE populations reflect explicit removal of the excluded frame from density calculations.

CARE and WSM TRAIN/VAL profile isolation remained unchanged: zero profile overlap in every fold.

## 6. Scientific interpretation and limitations

Policy C is conservative because it removes only the untrusted terminal interval and retains valid interior observations. It does not claim that historical openSMILE terminal semantics were recovered. The narrow limit is tied to the measured 16 kHz one-sample anomaly and is not a general boundary relaxation.

The validator still does not prove MantisV2 compatibility or performance. It also does not create real temporal caches. Any future cache builder must carry the exclusion mask and reason, retain original window times, and ensure excluded observations cannot enter Mantis inputs.

The analysis used only TRAIN/VAL data. No audio decoding, re-extraction, Mantis installation, checkpoint download, inference, resampling, cache generation, training, or TEST access occurred.

## 7. Reproduction

Strict mode:

```bash
python3 scripts/validate_mantis_egemaps_temporalization.py --check --boundary-policy strict
```

Policy C mode:

```bash
python3 scripts/validate_mantis_egemaps_temporalization.py --check --boundary-policy terminal-exclude
```

Both commands run synthetic checks first and emit aggregate JSON. Strict is expected to return nonzero because the historical CARE anomalies remain visible. Policy C returns zero only when the narrow exclusion rule resolves all boundary anomalies and every other integrity check passes.

## 8. Next gate

The next atomic step is a separately authorized MantisV2 package/checkpoint setup and synthetic tensor test. Any real temporal cache generation or Mantis inference remains a separate approval gate.

# Audio feature extraction audit summary

This anonymized summary was generated from the live CARE/WSM extracted trees,
the canonical split metadata, and the audited input-feature availability
manifest. It does not contain sample IDs, profile IDs, video IDs, or filesystem
paths.

## Output coverage

| Corpus | Video directories with incomplete output | Missing output files | Manifest rows without source VIDEO directory |
| --- | ---: | ---: | ---: |
| CARE | 101 | 1,515 | 0 |
| WSM | 87 | 1,350 | 3 |

Each affected source video is missing the complete `audio_id1/id2/id4` × five
fold set. No non-empty existing output file was classified as missing or
invalid; zero-byte output count was zero in both corpora.

The expected previous-audit counts are unchanged: 101 CARE videos and 87 WSM
videos with incomplete output. The three additional WSM manifest rows are a
separate category because their source VIDEO directories do not exist.

The complete-case reason is now evaluated only against the seven families in
the extraction `cohort_feature_family`. Problems in the other four families
remain listed in the detailed TSV as `additional_information` and do not
justify complete-case exclusion.

## Input-family problems among affected records

Counts are family/status occurrences, not unique videos; one video can appear
in multiple rows. The three source-missing WSM manifest rows contribute one
`MISSING` occurrence for each of the 11 families.

| Corpus | Input family | Manifest status | Occurrences |
| --- | --- | --- | ---: |
| CARE | `disvoice_articulation_static` | `INVALID_DTYPE` | 11 |
| CARE | `disvoice_phonation_static` | `INVALID_DTYPE` | 93 |
| CARE | `disvoice_prosody_dynamic` | `INVALID_DTYPE` | 27 |
| CARE | `disvoice_prosody_static` | `INVALID_DTYPE` | 95 |
| WSM | `disvoice_articulation_static` | `INVALID_DTYPE` | 45 |
| WSM | `disvoice_articulation_static` | `MISSING` | 3 |
| WSM | `disvoice_phonation_dynamic` | `EMPTY` | 85 |
| WSM | `disvoice_phonation_dynamic` | `MISSING` | 3 |
| WSM | `disvoice_phonation_static` | `INVALID_DTYPE` | 86 |
| WSM | `disvoice_phonation_static` | `MISSING` | 3 |
| WSM | `disvoice_prosody_dynamic` | `INVALID_DTYPE` | 76 |
| WSM | `disvoice_prosody_dynamic` | `MISSING` | 3 |
| WSM | `disvoice_prosody_static` | `INVALID_DTYPE` | 11 |
| WSM | `disvoice_prosody_static` | `MISSING` | 3 |
| WSM | `egemaps_func` | `INVALID_DTYPE` | 1 |
| WSM | `egemaps_func` | `MISSING` | 3 |
| WSM | `egemaps_lld` | `INVALID_DTYPE` | 1 |
| WSM | `egemaps_lld` | `MISSING` | 3 |
| WSM | `emotionmsp_embeds` | `MISSING` | 3 |
| WSM | `emotionmsp_trajectories` | `MISSING` | 3 |
| WSM | `trillsson4_1` | `MISSING` | 3 |
| WSM | `wav2vec_xlsr300m` | `MISSING` | 3 |

## Complete-case interpretation

- CARE: 101/101 affected videos have a non-`VALID` family in the seven-family
  cohort; complete-case exclusion is supported for all 101. The count is
  unchanged, but the reason is now cohort-scoped.
- WSM source-backed videos: 87/87 have a non-`VALID` family in the
  seven-family cohort; complete-case exclusion is supported for all 87. The
  count is unchanged, but the reason is now cohort-scoped.
- WSM manifest rows without a source VIDEO directory: 3, separately reported;
  all 11 input families are `MISSING` for each row, but the reason remains
  `не установлена: исходная VIDEO-папка отсутствует`.
- Unexplained output gaps among source-backed affected videos: 0 CARE, 0 WSM.

The detailed owner-only report is generated locally as
`reports/audio_feature_extraction_audit/missing_videos.tsv` and is excluded
from the public commit.

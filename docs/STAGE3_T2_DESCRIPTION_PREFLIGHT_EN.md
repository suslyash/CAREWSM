# Stage 3 T2 Observable-Description Preflight

Task: TASK-003D. Branch: `codex/task-003d`.

## Frozen generator and prompt

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Requested and resolved revision: `1dd1e02d981403da25ed73d43430e4ef598cb94b`
- Processor/model: `Qwen3VLProcessor` / `Qwen3VLForConditionalGeneration`
- Safetensors index SHA256: `520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070`
- Shards: `model-00001-of-00004.safetensors` `d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa`; `model-00002-of-00004.safetensors` `8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5`; `model-00003-of-00004.safetensors` `83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192`; `model-00004-of-00004.safetensors` `0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5`.
- Prompt SHA256: `118b93ba090af2b4ce755d9da1476e073cab0624760e9aa4c347faad8750ee4d`.
- System prompt: none.
- Generation: `do_sample=false`, `max_new_tokens=120`, one sequence, `num_beams=1`, no temperature/top-p tuning or prompt variants.
- Runtime correction after default video processing OOM: uniform deterministic `num_frames=8`, with `fps=null`; this setting is frozen and recorded in the external report.

## Environment and source audit

Package versions: torch `2.10.0`, transformers `5.14.1`, opencv-python `5.0.0.93`, accelerate `1.14.0`, safetensors `0.8.0`, Pillow `12.3.0`, huggingface-hub `1.26.0`. Device `cuda:0`, dtype `torch.bfloat16`; peak allocated/reserved GPU memory `14653126144/15919480832` bytes.

Canonical `build_manifest` segment identities were deterministically mapped to `<data-root>/<corpus>/<split>_labels/<video_id>/segments/<segment_file>`. Structural audit covered 8,622 `.mp4` rows: depression TRAIN 3,660, DEV 621, TEST 827; Parkinson TRAIN 2,665, DEV 312, TEST 537. All 8,622 existed and decoded/opened successfully; zero failures. TEST was structurally audited only and no TEST content was generated or manually inspected.

## Deterministic TRAIN/DEV sample

Stable SHA256 ordering of `corpus|split|video_id|segment_file`, six rows per stratum:

1. `depression|train|yWTjBnkbvQ8|yWTjBnkbvQ8_001.mp4`
2. `depression|train|oC3yAOC-6bU|oC3yAOC-6bU_011.mp4`
3. `depression|train|9eGROzF5OXE|9eGROzF5OXE_002.mp4`
4. `depression|train|IYe5yjc1jyM|IYe5yjc1jyM_012.mp4`
5. `depression|train|axHt0uSC_TU|axHt0uSC_TU_019.mp4`
6. `depression|train|s7u-xAWWyK4|s7u-xAWWyK4_025.mp4`
7. `depression|dev|pRPkme7EmZE|pRPkme7EmZE_002.mp4`
8. `depression|dev|QIS5gMTnCoo|QIS5gMTnCoo_008.mp4`
9. `depression|dev|PX7FZZISrCM|PX7FZZISrCM_005.mp4`
10. `depression|dev|FA6f_jTduYE|FA6f_jTduYE_002.mp4`
11. `depression|dev|3s14gCKn-yA|3s14gCKn-yA_001.mp4`
12. `depression|dev|NlzDZ2zrBGg|NlzDZ2zrBGg_003.mp4`
13. `parkinson|train|-1N99QsfWRk|-1N99QsfWRk_003.mp4`
14. `parkinson|train|VrlNc0BxP6o|VrlNc0BxP6o_006.mp4`
15. `parkinson|train|gkzaeZccedw|gkzaeZccedw_027.mp4`
16. `parkinson|train|KGdlcLhLn3k|KGdlcLhLn3k_004.mp4`
17. `parkinson|train|_wzMMgvZ6tQ|_wzMMgvZ6tQ_003.mp4`
18. `parkinson|train|z5a4ybxov-M|z5a4ybxov-M_006.mp4`
19. `parkinson|dev|F61NqU2lnJc|F61NqU2lnJc_009.mp4`
20. `parkinson|dev|Bjk9_FzdK-Y|Bjk9_FzdK-Y_002.mp4`
21. `parkinson|dev|E9e_EOMbBqQ|E9e_EOMbBqQ_002.mp4`
22. `parkinson|dev|qoDeiiR4Omk|qoDeiiR4Omk_006.mp4`
23. `parkinson|dev|LWPKg6J_Iig|LWPKg6J_Iig_006.mp4`
24. `parkinson|dev|4-d_F3MczUE|4-d_F3MczUE_005.mp4`

All 24 outputs were nonempty and judged visually grounded/mostly observable. Manual aggregate flags: major unsupported/hallucinated fact 0; diagnosis/disease/health-state inference 0; demographic/identity inference 4; causal/medication inference 0; dataset/task/label leakage 0; concise enough for semantic encoding 18/24. The four demographic/identity failures were explicit gender terms; six concision failures were overly long, repetitive, or truncated outputs. No disease labels were used to judge description quality.

The lexicographically first sampled item was generated twice from fresh calls. Exact equality: yes. Hash A and B: `bcad9ba7c4a55880b82e0106afcb93ebc3476a902683b9ea60725468b5aa63da` and the same hash B.

## Gate and artifact

External report: `/media/maxim/Programs/Features/WSM/stage3_t2_preflight/qwen3vl8b_observable_preflight_v1.json`.

Report SHA256: `767d687e2f857938fa882555250d7a4be4159556e49537377967d82c36407930`.

Exact result: `T2 DESCRIPTION GENERATION CONTRACT BLOCKED`.

Failed conditions: condition 5 (zero demographic/identity inference) and condition 9 (at least 22/24 concise enough for semantic encoding; observed 18/24). The frozen model and prompt were not changed to remediate the block. No training, description cache, T2 implementation, performance metric, or TEST generation was performed.

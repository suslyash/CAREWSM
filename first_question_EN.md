# Original Project Brief

Develop a novel multimodal, multitask, multilabel method for binary Parkinson's disease and binary depression recognition. Planned modalities: audio, video, transcript text, and descriptions or features from an LLM/VLM such as Qwen3-Omni-30B-A3B-Instruct. The target is an A* conference or Q1 journal.

The Parkinson and depression datasets are disjoint corpora. Each is annotated for only its own task, unlike standard jointly annotated MTL.

The starting project uses [chimera_ml](https://github.com/markitantov/chimera_ml) for multitask acoustic recognition.

## Project Requirements

1. Implement within chimera_ml and follow current src/audio style.
2. Separate unimodal and multimodal source by package: src/audio, src/video, src/text, src/fusion, etc. Use one shared experiment_name with modality-specific experiments underneath; fundamentally different global approaches are exceptions.
3. Put additional executable scripts under scripts/<modality>, not src.
4. Register every selectable component.
5. Required components: checkpoint_callback, snapshot_callback, early_stopping_callback, console_file_logger, mlflow_logger, wsm_summary_callback, plus an integral metric callback based on wsm_segment_callback when useful.
6. Put universal callbacks/losses/etc. under common.
7. Select on DEV only:

\[
Mean\_Score=(Score_D+Score_P)/2,
\]

\[
Score_D=(UAR_D+MF1_D)/2,\qquad
Score_P=(UAR_P+MF1_P)/2.
\]

Report TEST_NONE, TEST_SOFT, and TEST_HARD only for information and paper comparison, including DEPART.

8. Limit unimodal work: at most two video and two text experiments. Implement a DEPART-like video model. Use the existing src/audio method unchanged as the audio baseline.
9. Focus on the multimodal multitask multilabel method.
10. Unit tests are optional.

## Requested Deliverables

1. Analyze docs/2402.19078v3.pdf, docs/2501.10945v3.pdf, docs/BDCC-10-00089.pdf, and docs/Ryumin_EMNLP.pdf; summarize them in docs/BASELINES.md.
2. Analyze docs/SOTA_REVIEW.md.
3. Document the initial solution/structure in docs/PROJECT_INIT_STRUCTURE.md and formalize requirements in docs/PROJECT_REQUIREMENTS.md.
4. Use those documents to create a staged development and experiment plan in docs/PLAN.md.
5. Create AGENTS.md that makes Codex follow the plan strictly.
6. Structure the workflow for manager orchestration: Codex result → manager analysis plus next task file → Codex result → repeat.


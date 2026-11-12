# audio_turns_combined Program Design

This branch implements the bounded static recording-level research program authorized by the manager task. It preserves the four audited families and fixed joint CARE+WSM five-fold protocol.

## Representation contract

Families are ordered A=`emotionmsp_embeds` (1024), B=`emotionmsp_trajectories` (3), C=`trillsson4_1` (1024), D=`wav2vec_xlsr300m` (1024). R1 is per-family arithmetic mean over turns. R2 is per-family population mean followed by population standard deviation, concatenated in family order. The live validation passed for dimension 6150 on ABCD mean+std, finite values, aligned turn IDs, singleton zero standard deviation, and no checked TRAIN/VAL profile overlap.

The four NPZ families are turn-level artifacts. The NPZ `mean` channel is used for TRILLsson and XLS-R before turn pooling; their stored NPZ `std` channel is not mixed into R2. No sequence model, temporal attention, recurrent layer, positional encoding, or repeated static timeline is used.

## Bounded stages

- Stage 1: 15 subsets × 5 folds = 75 static MLP mean runs.
- Stage 2: up to 4 selected subsets × 5 folds = 20 static gated-fusion mean runs.
- Stage 3: up to 4 selected configurations × 5 folds = 20 mean+std runs.
- Stage 4: two alternatives × 5 folds = 10 tuning runs.
- Stage 5: Equal, STCH, PCGrad, and DB-MTL × 5 folds = 20 runs.
- Maximum: 145 runs.

All checkpoint and candidate decisions use `val_mean_corpora_tasks/macro_recall`; TEST is descriptive only. The master launcher persists failures and continues to independent runs.

## Live-data note

The current CARE root has the authoritative `train.csv`, `dev.csv`, `test.csv`, and fold files but does not contain `splits/videos.csv`; WSM does contain it. The active DataModule joins the availability manifest to authoritative split metadata and does not require the absent convenience file. No source split or feature artifact was modified.

## Launch state

Preflight passed locally. The platform blocked autonomous GPU launch as a high-impact action, so no run, checkpoint, MLflow training record, or fold-0 result is claimed by this handoff.

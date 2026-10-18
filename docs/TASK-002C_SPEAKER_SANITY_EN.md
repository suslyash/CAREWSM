# TASK-002C — bounded WSM speaker sanity and listening gate

## Outcome

This task prepares a reproducible owner listening check for the same three WSM
TRAIN recordings used by TASK-002B. No new media were selected, exported,
downloaded, or modified. No inference was rerun. The intervals below are
short windows around DiariZen-reported speech segments; they are model output,
not human speaker labels or ground truth.

The owner should listen to each recording's early, middle, and late window and
record whether the audible material contains one speaker, two or more speakers,
or is uncertain. The owner must separately record whether respondent identity is
identifiable from the audio (`yes`, `no`, or `uncertain`).

## Fixed recordings and model observations

| sample ID | source path | source duration (s) | DiariZen speakers | DiariZen segments | longest-speaker coverage |
|---|---|---:|---:|---:|---:|
| `DEPRESSION__DEPRESSION__-7UpRmNVJzQ__-7UpRmNVJzQ_001` | `/media/maxim/Databases/WSM_NEW/depression/train_labels/-7UpRmNVJzQ/segments/-7UpRmNVJzQ_001.mp4` | 75.263965 | 1 | 23 | 0.659545 |
| `CONTROL__DEPRESSION__0efS0axfX0E__0efS0axfX0E_001` | `/media/maxim/Databases/WSM_NEW/depression/train_labels/0efS0axfX0E/segments/0efS0axfX0E_001.mp4` | 61.463965 | 1 | 24 | 0.587987 |
| `PARKINSON__PARKINSON__-1N99QsfWRk__-1N99QsfWRk_001` | `/media/maxim/Databases/WSM_NEW/parkinson/train_labels/-1N99QsfWRk/segments/-1N99QsfWRk_001.mp4` | 100.063965 | 1 | 21 | 0.871842 |

All three source files were previously measured as AAC, 16 kHz, two-channel
media. TASK-002B found no evidence that the stereo channels were separate
respondent/interviewer channels. The longest-speaker mapping is therefore only
a diagnostic and does not identify the respondent.

## Owner playback commands

These commands use `ffplay` audio-only playback. Each window is at most 10
seconds, and each command reads the original file without writing a derivative.
The `-ss` values are chosen from observed DiariZen segment locations so that
speech is likely, but human confirmation is still required.

### DEPRESSION recording

```bash
ffplay -nodisp -autoexit -loglevel warning -vn -ss 0.30 -t 5.0   /media/maxim/Databases/WSM_NEW/depression/train_labels/-7UpRmNVJzQ/segments/-7UpRmNVJzQ_001.mp4

ffplay -nodisp -autoexit -loglevel warning -vn -ss 13.80 -t 6.0   /media/maxim/Databases/WSM_NEW/depression/train_labels/-7UpRmNVJzQ/segments/-7UpRmNVJzQ_001.mp4

ffplay -nodisp -autoexit -loglevel warning -vn -ss 53.00 -t 8.0   /media/maxim/Databases/WSM_NEW/depression/train_labels/-7UpRmNVJzQ/segments/-7UpRmNVJzQ_001.mp4
```

Model-reported reference segments: approximately `0.2925–2.2325`,
`13.8125–17.7525`, and `52.9925–60.6525` seconds.

### CONTROL recording

```bash
ffplay -nodisp -autoexit -loglevel warning -vn -ss 1.00 -t 5.0   /media/maxim/Databases/WSM_NEW/depression/train_labels/0efS0axfX0E/segments/0efS0axfX0E_001.mp4

ffplay -nodisp -autoexit -loglevel warning -vn -ss 31.50 -t 6.0   /media/maxim/Databases/WSM_NEW/depression/train_labels/0efS0axfX0E/segments/0efS0axfX0E_001.mp4

ffplay -nodisp -autoexit -loglevel warning -vn -ss 53.40 -t 3.0   /media/maxim/Databases/WSM_NEW/depression/train_labels/0efS0axfX0E/segments/0efS0axfX0E_001.mp4
```

Model-reported reference segments: approximately `0.9925–2.3925`,
`31.4725–35.7925`, and `53.3925–56.1325` seconds.

### PARKINSON recording

```bash
ffplay -nodisp -autoexit -loglevel warning -vn -ss 1.40 -t 9.0   /media/maxim/Databases/WSM_NEW/parkinson/train_labels/-1N99QsfWRk/segments/-1N99QsfWRk_001.mp4

ffplay -nodisp -autoexit -loglevel warning -vn -ss 46.55 -t 5.0   /media/maxim/Databases/WSM_NEW/parkinson/train_labels/-1N99QsfWRk/segments/-1N99QsfWRk_001.mp4

ffplay -nodisp -autoexit -loglevel warning -vn -ss 90.60 -t 5.0   /media/maxim/Databases/WSM_NEW/parkinson/train_labels/-1N99QsfWRk/segments/-1N99QsfWRk_001.mp4
```

Model-reported reference segments: approximately `1.4325–9.6325`,
`46.5525–50.8525`, and `90.6325–95.2725` seconds.

If `ffplay` is unavailable, the owner may use an already-installed local audio
player with the same source file and time windows. Do not install a player or
export clips for this task.

## Listening checklist

For each window, listen for:

1. a continuous human voice versus silence, music, or noise;
2. a second voice entering, overlapping, or responding;
3. changes in voice identity, pitch, accent, microphone position, or channel
   character that suggest a second speaker;
4. whether the audible speaker can be identified as respondent or interviewer
   from the recording context alone.

A one-speaker DiariZen result means only that the model emitted one speaker
label. It does not prove that an interviewer is absent, merged, silent, or
correctly assigned to the respondent.

## Owner response protocol

Please return one row per recording using this format:

| sample ID | human audible speakers: one / two-or-more / uncertain | respondent identifiable: yes / no / uncertain | brief evidence |
|---|---|---|---|
| DEPRESSION__... |  |  |  |
| CONTROL__... |  |  |  |
| PARKINSON__... |  |  |  |

Until the owner supplies these observations, manual speaker confirmation is
**pending**. No DiariZen retain/drop decision and no participant-turn quality
claim is made from the automatic one-speaker output alone.

## Scope and safety

No TEST or DEV media were accessed. No additional WSM media were enumerated.
No downloads, installations, inference reruns, tests, extraction, training, or
writes under CARE/WSM were performed for TASK-002C.

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WAV files for WSM video segments.")
    parser.add_argument("--data-root", default="/media/maxim/Databases/WSM_NEW")
    parser.add_argument("--task", default="all")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    tasks = ("depression", "parkinson") if args.task == "all" else (args.task,)
    rows: list[dict[str, object]] = []
    skipped_empty = 0
    for task in tasks:
        for split in ("train", "dev", "test"):
            labels = pd.read_csv(data_root / task / "other_labels" / f"{split}_labels_segments_min.csv")
            labels.columns = [str(column).strip() for column in labels.columns]
            skipped_empty += int(labels["segment_file"].isna().sum())
            labels = labels[labels["segment_file"].notna()].copy()
            for item in labels.to_dict("records"):
                video_id = str(item["video_id"])
                segment_file = str(item["segment_file"])
                rows.append(
                    {
                        "task": task,
                        "split": split,
                        "video_id": video_id,
                        "segment_file": segment_file,
                        "segment_path": str(data_root / task / f"{split}_labels" / video_id / "segments" / segment_file),
                    }
                )

    records = pd.DataFrame(rows)
    if args.limit is not None:
        records = records.head(int(args.limit)).copy()

    saved = 0
    skipped = 0
    failed: list[dict[str, object]] = []
    failed_log = data_root / ("failed_segment_wavs.csv" if args.task == "all" else f"failed_segment_wavs_{args.task}.csv")

    for row in tqdm(records.itertuples(index=False), total=len(records), desc="extract segment wavs"):
        out_path = Path(row.segment_path).with_suffix(".wav")
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(row.segment_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(args.sample_rate),
            str(out_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            out_path.unlink(missing_ok=True)
            failed.append(
                {
                    "task": row.task,
                    "split": row.split,
                    "video_id": row.video_id,
                    "segment_file": row.segment_file,
                    "segment_path": row.segment_path,
                    "wav_path": str(out_path),
                    "returncode": exc.returncode,
                    "stderr": (exc.stderr or "").strip().replace("\n", " | "),
                }
            )
            continue
        saved += 1

    if failed:
        with failed_log.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(failed[0].keys()))
            writer.writeheader()
            writer.writerows(failed)

    print(f"saved={saved} skipped={skipped} skipped_empty={skipped_empty} failed={len(failed)}")
    if failed:
        print(f"failed_log={failed_log}")


if __name__ == "__main__":
    main()

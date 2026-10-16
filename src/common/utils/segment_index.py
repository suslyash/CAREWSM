from __future__ import annotations

from pathlib import Path

import pandas as pd


TASKS = ("depression", "parkinson")
TASK_TO_ID = {"depression": 0, "parkinson": 1}
SPLIT_PRIORITY = {"train": 0, "dev": 1, "test": 2}
BAD_SEGMENTS = {
    ("depression", "dev", "3s14gCKn-yA", "3s14gCKn-yA_005.mp4"),
    ("depression", "dev", "49LbNLMWZwM", "49LbNLMWZwM_001.mp4"),
    ("depression", "dev", "4QIa1kSG45A", "4QIa1kSG45A_001.mp4"),
}


def build_wsm_segment_index(
    data_root: str | Path = "/media/maxim/Databases/WSM_NEW",
    task: str = "depression",
) -> pd.DataFrame:
    data_root = Path(data_root)
    rows: list[dict[str, object]] = []

    for split in ("train", "dev", "test"):
        suffix = "_mishas" if split == "test" else ""
        csv_path = data_root / task / f"{split}_labels_segments_min_filtered{suffix}.csv"
        labels = pd.read_csv(csv_path, sep=";" if suffix else ",")
        labels.columns = [str(column).strip() for column in labels.columns]

        for item in labels.to_dict("records"):
            video_id = str(item["video_id"])
            segment_file = str(item["segment_file"])
            if (task, split, video_id, segment_file) in BAD_SEGMENTS:
                continue

            video_dir = data_root / task / f"{split}_labels" / video_id
            rows.append(
                {
                    "task": task,
                    "task_id": TASK_TO_ID[task],
                    "split": split,
                    "source_split": split,
                    "video_id": video_id,
                    "segment_file": segment_file,
                    "label": int(item["diagnosis"]),
                    "segment_path": str(video_dir / "segments" / segment_file),
                    "soft_filter": item.get("soft_filter"),
                    "hard_filter": item.get("hard_filter"),
                }
            )

    records = pd.DataFrame(rows)
    return records.sort_values(["split", "video_id", "segment_file"]).reset_index(drop=True)


def build_wsm_multitask_segment_index(data_root: str | Path = "/media/maxim/Databases/WSM_NEW") -> pd.DataFrame:
    records = pd.concat(
        [build_wsm_segment_index(data_root, task=task) for task in TASKS],
        ignore_index=True,
    )
    split_by_video: dict[str, str] = {}
    for row in records.itertuples(index=False):
        current = split_by_video.get(row.video_id)
        if current is None or SPLIT_PRIORITY[row.source_split] > SPLIT_PRIORITY[current]:
            split_by_video[row.video_id] = row.source_split

    records["split"] = records["video_id"].map(split_by_video)
    records = records[records["split"] == records["source_split"]].copy()
    return records.sort_values(["split", "task", "video_id", "segment_file"]).reset_index(drop=True)

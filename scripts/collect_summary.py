"""Collect complete CAREWSM audio run summaries without modifying source logs."""
from __future__ import annotations

import argparse
from pathlib import Path


def collect(root: Path, output: Path) -> int:
    runs = []
    for summary in sorted(root.glob("**/summary.txt")):
        run_dir = summary.parent
        configs = sorted((*run_dir.glob("*.yaml"), *run_dir.glob("*.yml")))
        if not configs or not (run_dir / "train.log").is_file():
            continue
        runs.append((run_dir, configs[0], summary))
    with output.open("w", encoding="utf-8") as handle:
        for run_dir, config, summary in runs:
            handle.write(f"EXPERIMENT: {run_dir.name}\n")
            handle.write(f"RUN: {run_dir}\nCONFIG: {config}\n")
            handle.write(config.read_text(encoding="utf-8").rstrip() + "\nRESULTS:\n")
            handle.write(summary.read_text(encoding="utf-8").rstrip() + "\n\n")
    return len(runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("logs/audio"))
    parser.add_argument("--output", type=Path, default=Path("all_summary.txt"))
    args = parser.parse_args()
    count = collect(args.root, args.output)
    print(f"Saved {count} complete runs to {args.output.resolve()}")


if __name__ == "__main__":
    main()

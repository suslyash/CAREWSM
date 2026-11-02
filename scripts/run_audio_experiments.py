"""Launch explicitly selected CAREWSM audio YAML series through Chimera-ML."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

SERIES = {
    "mptsnet_native_egemaps_lld": Path("configs/audio/series/audio_mptsnet_native_egemaps_lld.yaml"),
    "mdformer_native_egemaps_lld": Path("configs/audio/series/audio_mdformer_native_egemaps_lld.yaml"),
}


def list_series() -> None:
    for name, path in SERIES.items():
        print(f"{name} -> {path}")


def fold_config(path: Path, fold: int) -> tuple[tempfile.NamedTemporaryFile, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_name = str(config["experiment_info"]["params"]["experiment_name"]).replace("fold0", f"fold{fold}")
    config["data"]["params"]["fold"] = fold
    config["experiment_info"]["params"]["experiment_name"] = base_name
    config["experiment_info"]["params"]["run_name"] = f"fold{fold}"
    target = Path("logs") / base_name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {target}")
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(config, handle, sort_keys=False)
    handle.flush()
    return handle, base_name


def run_series(name: str) -> None:
    source = SERIES[name]
    for fold in range(5):
        handle, run_name = fold_config(source, fold)
        try:
            print(f"START {run_name}", flush=True)
            subprocess.run([".venv/bin/chimera-ml", "train", "-c", handle.name], check=True)
            print(f"DONE {run_name}", flush=True)
        finally:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="list available series without launching")
    group.add_argument("--series", choices=sorted(SERIES), help="launch one series across folds 0-4")
    group.add_argument("--all", action="store_true", help="launch all explicitly registered series")
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args()
    if args.list:
        list_series()
        return 0
    selected = list(SERIES) if args.all else [args.series]
    if not selected or selected == [None]:
        parser.error("choose --list, --series NAME, or --all")
    for name in selected:
        run_series(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

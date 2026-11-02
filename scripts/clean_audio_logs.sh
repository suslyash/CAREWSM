#!/usr/bin/env bash
set -euo pipefail

root="logs"
find "$root" -mindepth 1 -maxdepth 1 -type d -name 'audio_*_fold*' -print
if [[ "${1:-}" == "--apply" ]]; then
  while IFS= read -r directory; do
    rm -rf -- "$directory"
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -name 'audio_*_fold*' -print)
fi

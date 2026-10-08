#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: run_v3_queue.sh GPU_ID QUEUE_TSV" >&2
  exit 2
fi

gpu_id="$1"
queue_path="$2"
export CUDA_VISIBLE_DEVICES="$gpu_id"

while IFS=$'\t' read -r data_config model_config seed output_dir extra; do
  [[ -z "${data_config}" || "${data_config:0:1}" == "#" ]] && continue
  overrides=()
  if [[ -n "${extra:-}" ]]; then
    read -r -a overrides <<< "$extra"
  fi
  scripts/run_v3_job.sh "$data_config" "$model_config" "$seed" "$output_dir" "${overrides[@]}"
done < "$queue_path"

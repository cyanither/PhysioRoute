#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: run_v3_queue_chain.sh GPU_ID QUEUE_TSV [QUEUE_TSV...]" >&2
  exit 2
fi

gpu_id="$1"
shift

for queue in "$@"; do
  echo "starting queue: gpu=${gpu_id} queue=${queue}"
  scripts/run_v3_queue.sh "$gpu_id" "$queue"
  echo "finished queue: gpu=${gpu_id} queue=${queue}"
done

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: run_v3_job.sh DATA_CONFIG MODEL_CONFIG SEED OUTPUT_DIR [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

data_config="$1"
model_config="$2"
seed="$3"
output_dir="$4"
shift 4

mkdir -p "$output_dir" logs/v3_runs
slug="${output_dir//\//__}"
log_path="logs/v3_runs/${slug}.log"

if [[ -s "$output_dir/COMPLETE" && -s "$output_dir/test_metrics.json" && -s "$output_dir/predictions.npz" ]]; then
  echo "already complete: $output_dir"
  exit 0
fi

env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
  .venv/bin/python -m src.train \
  "data=${data_config}" "model=${model_config}" "seed=${seed}" \
  epochs=40 patience=8 batch_size=64 num_workers=0 tensorboard.enabled=false \
  "output_dir=${output_dir}" "$@" 2>&1 | tee "$log_path"

test -s "$output_dir/best.pt"
test -s "$output_dir/metrics.json"
test -s "$output_dir/config.yaml"
test -s "$output_dir/source_manifest.json"

env PYTHONUNBUFFERED=1 .venv/bin/python -m src.eval \
  "data=${data_config}" "model=${model_config}" "seed=${seed}" \
  batch_size=256 num_workers=0 "output_dir=${output_dir}" "$@" 2>&1 | tee -a "$log_path"

.venv/bin/python - "$output_dir" <<'PY'
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
train = json.loads((root / "metrics.json").read_text())
test = json.loads((root / "test_metrics.json").read_text())
if not train.get("history"):
    raise SystemExit("empty training history")
if "hr_mae" not in test or not math.isfinite(float(test["hr_mae"])):
    raise SystemExit("missing or nonfinite hr_mae")
if not (root / "predictions.npz").is_file():
    raise SystemExit("missing predictions.npz")
status = {
    "status": "complete",
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "epochs_ran": len(train["history"]),
    "best_val_loss": train["best_val_loss"],
    "test_metrics": test,
}
(root / "run_status.json").write_text(json.dumps(status, indent=2))
(root / "COMPLETE").write_text(status["completed_utc"] + "\n")
PY

echo "complete: $output_dir"

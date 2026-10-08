#!/usr/bin/env python3
"""Create a read-only HR-only PTT-PPG view for mixed-dataset transfer."""

from __future__ import annotations

import json
import os
from pathlib import Path


SOURCE = Path("data/processed_v3/ptt_ppg_h8")
OUTPUT = Path("data/processed_v3/ptt_ppg_h8_transfer")


def link(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    metadata["transfer_view"] = {
        "source": str(SOURCE.resolve()),
        "purpose": "HR-only mixed-dataset collation",
        "omitted_arrays": ["y_pat.npy", "current_pat.npy"],
        "signal_and_hr_arrays_modified": False,
    }
    metadata["targets"] = ["hr"]
    metadata["input_only_baseline_metadata"] = ["current_hr"]
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for split in ("train", "val", "test"):
        output_dir = OUTPUT / split
        output_dir.mkdir()
        for source in sorted((SOURCE / split).glob("*.npy")):
            if source.name in {"y_pat.npy", "current_pat.npy"}:
                continue
            link(source, output_dir / source.name)
    required = {
        "X.npy", "y_hr.npy", "subject.npy", "record.npy", "start_idx.npy",
        "current_hr.npy", "activity.npy", "future_activity.npy", "activity_valid.npy",
    }
    for split in ("train", "val", "test"):
        names = {path.name for path in (OUTPUT / split).glob("*.npy")}
        if names != required:
            raise RuntimeError(f"Unexpected {split} transfer arrays: {sorted(names)}")
    print(json.dumps({"status": "PASS", "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()

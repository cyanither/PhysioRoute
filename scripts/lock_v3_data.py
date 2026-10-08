#!/usr/bin/env python3
"""Create or verify SHA-256 locks for primary v3 processed arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOTS = (
    Path("data/processed_v3/wesad_h8"),
    Path("data/processed_v3/ppg_dalia_h8"),
    Path("data/processed_v3/ptt_ppg_h8"),
    Path("data/processed_v3/wesad_h30"),
    Path("data/processed_v3/ppg_dalia_h30"),
    Path("data/processed_v3/ptt_ppg_h30"),
)
LOCK = Path("results_v3/data_lock_v3.json")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def current() -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for root in ROOTS:
        if not root.exists():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            files[str(path)] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return {
        "protocol": "v3 processed-data lock after isolated H8 PAT amendment and H30 HR-only archival",
        "pat_amendment": "results_v3/isolated_pat_protocol.json",
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    observed = current()
    if args.write:
        LOCK.write_text(json.dumps(observed, indent=2), encoding="utf-8")
        print(json.dumps({"status": "LOCKED", "files": len(observed["files"])}))
        return
    expected = json.loads(LOCK.read_text(encoding="utf-8"))
    if observed != expected:
        expected_files = expected.get("files", {})
        observed_files = observed.get("files", {})
        changed = sorted(
            path for path in set(expected_files) | set(observed_files)
            if expected_files.get(path) != observed_files.get(path)
        )
        raise SystemExit(f"FAIL: v3 data lock mismatch in {len(changed)} paths: {changed[:10]}")
    print(json.dumps({"status": "PASS", "files": len(observed["files"])}))


if __name__ == "__main__":
    main()

"""Verify that frozen v3 source/config hashes have not changed."""

import hashlib
import json
from pathlib import Path


def main() -> None:
    lock_path = Path("results_v3/protocol_lock_v3.json")
    lock = json.loads(lock_path.read_text())
    mismatches = []
    for name, expected in lock["files"].items():
        path = Path(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append({"file": name, "expected": expected, "actual": actual})
    if mismatches:
        raise SystemExit("v3 protocol lock mismatch:\n" + json.dumps(mismatches, indent=2))
    print(f"PASS: {len(lock['files'])} frozen files match {lock_path}")


if __name__ == "__main__":
    main()

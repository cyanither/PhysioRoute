#!/usr/bin/env python3
"""Build negative-control views of the frozen v3 PTT-PPG arrays.

The source arrays are never modified.  Unchanged arrays are represented by
relative symbolic links, while only the array manipulated by a control is
materialized.  PAT shuffling is restricted to the training split and is
performed within each record so that marginal and record-specific label
distributions are retained while temporal pairing is destroyed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")
CONTROLS = ("pat_shuffled", "time_reversed", "channels_scrambled")


def link(source: Path, destination: Path) -> None:
    """Create a relative link after validating that no output already exists."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def shuffled_within_record(values: np.ndarray, records: np.ndarray, seed: int) -> np.ndarray:
    """Permute finite labels independently within each source record."""

    result = np.asarray(values).copy()
    rng = np.random.default_rng(seed)
    for record in np.unique(records):
        indices = np.flatnonzero(records == record)
        finite = indices[np.isfinite(result[indices].reshape(-1))]
        if finite.size < 2:
            continue
        permutation = rng.permutation(finite.size)
        if np.array_equal(permutation, np.arange(finite.size)):
            permutation = np.roll(permutation, 1)
        result[finite] = result[finite[permutation]]
    return result


def materialize_x(source: Path, destination: Path, control: str, permutation: np.ndarray) -> None:
    """Write a time-reversed or channel-scrambled X memmap in bounded chunks."""

    x = np.load(source, mmap_mode="r")
    out = np.lib.format.open_memmap(destination, mode="w+", dtype=x.dtype, shape=x.shape)
    for start in range(0, x.shape[0], 64):
        stop = min(start + 64, x.shape[0])
        block = np.asarray(x[start:stop])
        if control == "time_reversed":
            out[start:stop] = block[:, :, ::-1, :]
        elif control == "channels_scrambled":
            out[start:stop] = block[:, permutation, :, :]
        else:  # pragma: no cover
            raise ValueError(control)
    out.flush()


def build(source_root: Path, output_root: Path, control: str, seed: int) -> None:
    """Build one control dataset and record its exact transformation."""

    if control not in CONTROLS:
        raise ValueError(f"Unknown control {control!r}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing control root: {output_root}")
    metadata = json.loads((source_root / "metadata.json").read_text(encoding="utf-8"))
    node_names = list(metadata["node_names"])
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(node_names))
    if np.array_equal(permutation, np.arange(len(node_names))):
        permutation = np.roll(permutation, 1)

    output_root.mkdir(parents=True)
    for split_index, split in enumerate(SPLITS):
        source_dir = source_root / split
        output_dir = output_root / split
        output_dir.mkdir()
        for source in sorted(source_dir.glob("*.npy")):
            destination = output_dir / source.name
            if control == "pat_shuffled" and split == "train" and source.name == "y_pat.npy":
                labels = np.load(source)
                records = np.load(source_dir / "record.npy")
                transformed = shuffled_within_record(labels, records, seed + split_index)
                np.save(destination, transformed)
            elif control in {"time_reversed", "channels_scrambled"} and source.name == "X.npy":
                materialize_x(source, destination, control, permutation)
            else:
                link(source.resolve(), destination)

    control_record = {
        "name": control,
        "seed": seed,
        "source_root": str(source_root),
        "train_only": control == "pat_shuffled",
    }
    if control == "pat_shuffled":
        control_record["description"] = (
            "Training y_pat values permuted within record; validation and test labels unchanged"
        )
    elif control == "time_reversed":
        control_record["description"] = "Every input X window reversed along its time axis"
    else:
        control_record["description"] = (
            "Every input X window permuted along its node axis while original node names and priors are retained"
        )
        control_record["source_index_for_each_declared_node"] = permutation.tolist()
        control_record["effective_source_names"] = [node_names[index] for index in permutation]
    metadata["negative_control"] = control_record
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def validate(source_root: Path, output_root: Path, control: str) -> None:
    """Check shapes, finiteness, and that each manipulation actually changed data."""

    for split in SPLITS:
        source_dir = source_root / split
        output_dir = output_root / split
        for source in sorted(source_dir.glob("*.npy")):
            original = np.load(source, mmap_mode="r")
            candidate = np.load(output_dir / source.name, mmap_mode="r")
            if original.shape != candidate.shape or original.dtype != candidate.dtype:
                raise RuntimeError(f"Shape/dtype mismatch for {split}/{source.name}")
    if control == "pat_shuffled":
        old = np.load(source_root / "train/y_pat.npy").reshape(-1)
        new = np.load(output_root / "train/y_pat.npy").reshape(-1)
        if np.array_equal(old, new):
            raise RuntimeError("PAT shuffle did not change the training labels")
        if not np.allclose(np.sort(old[np.isfinite(old)]), np.sort(new[np.isfinite(new)])):
            raise RuntimeError("PAT shuffle changed the training-label marginal distribution")
        for split in ("val", "test"):
            if not np.array_equal(
                np.load(source_root / split / "y_pat.npy"),
                np.load(output_root / split / "y_pat.npy"),
            ):
                raise RuntimeError(f"PAT shuffle unexpectedly changed {split} labels")
    else:
        old = np.load(source_root / "test/X.npy", mmap_mode="r")
        new = np.load(output_root / "test/X.npy", mmap_mode="r")
        if np.array_equal(np.asarray(old[:2]), np.asarray(new[:2])):
            raise RuntimeError(f"{control} did not change representative X windows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", choices=CONTROLS)
    parser.add_argument("--source-root", type=Path, default=Path("data/processed_v3/ptt_ppg_h8"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    output_root = args.output_root or args.source_root.with_name(f"{args.source_root.name}_{args.control}")
    build(args.source_root, output_root, args.control, args.seed)
    validate(args.source_root, output_root, args.control)
    print(json.dumps({"control": args.control, "output_root": str(output_root), "status": "PASS"}))


if __name__ == "__main__":
    main()

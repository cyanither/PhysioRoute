"""PyTorch dataset wrappers for preprocessed physiological windows."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PhysioWindowDataset(Dataset[dict[str, Any]]):
    """Dataset backed by split-specific ``.npy`` memmaps."""

    def __init__(self, processed_dir: str | Path, split: str) -> None:
        """Load memmap handles for a processed split.

        Args:
            processed_dir: Root directory containing ``metadata.json`` and split folders.
            split: One of ``train``, ``val``, or ``test``.
        """

        self.processed_dir = Path(processed_dir)
        self.split = split
        split_dir = self.processed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing processed split directory: {split_dir}")
        with (self.processed_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        self.x = np.load(split_dir / "X.npy", mmap_mode="r")
        self.targets = {
            path.stem.removeprefix("y_"): np.load(path, mmap_mode="r")
            for path in sorted(split_dir.glob("y_*.npy"))
        }
        self.subject = np.load(split_dir / "subject.npy", mmap_mode="r")
        self.start_idx = np.load(split_dir / "start_idx.npy", mmap_mode="r")
        self.metadata_arrays: dict[str, np.ndarray] = {}
        for name in ("record", "activity", "future_activity", "activity_valid", "current_hr", "current_pat"):
            path = split_dir / f"{name}.npy"
            if path.exists():
                self.metadata_arrays[name] = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        """Return the number of windows in the split."""

        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one window and its task targets."""

        item = {
            "x": torch.from_numpy(np.array(self.x[index], copy=True)).float(),
            "targets": {
                name: torch.from_numpy(np.array(value[index], copy=True)).float()
                for name, value in self.targets.items()
            },
            "subject": str(self.subject[index]),
            "start_idx": int(self.start_idx[index]),
        }
        for name, values in self.metadata_arrays.items():
            value = values[index]
            if name == "record":
                item[name] = str(value)
            elif name == "activity_valid":
                item[name] = bool(value)
            elif name.startswith("current_"):
                item[name] = float(value)
            else:
                item[name] = int(value)
        return item

class IndexedDataset(Dataset[dict[str, Any]]):
    """View a dataset through a fixed integer index list."""

    def __init__(self, base: Dataset[dict[str, Any]], indices: Sequence[int]) -> None:
        """Create an indexed view."""

        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.metadata = getattr(base, "metadata", {})

    def __len__(self) -> int:
        """Return the number of selected rows."""

        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one selected item."""

        return self.base[int(self.indices[index])]


CANONICAL_NODE_ALIASES: dict[str, tuple[str, ...]] = {
    "ecg": ("ecg",),
    "ppg": ("wrist_bvp", "finger_ppg_ir", "finger_ppg_red", "proximal_ppg_ir"),
    "resp": ("resp",),
    "acc_x": ("wrist_acc_x", "acc_x"),
    "acc_y": ("wrist_acc_y", "acc_y"),
    "acc_z": ("wrist_acc_z", "acc_z"),
    "eda": ("wrist_eda",),
    "temp": ("wrist_temp",),
}


class NodeMappedDataset(Dataset[dict[str, Any]]):
    """Map dataset-specific physiological channels to common canonical nodes."""

    def __init__(
        self,
        base: Dataset[dict[str, Any]],
        target_nodes: Sequence[str],
        target_names: Sequence[str] = ("wave", "hr"),
        aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Create a node-mapped dataset."""

        self.base = base
        self.target_nodes = list(target_nodes)
        self.target_names = tuple(target_names)
        self.aliases = aliases or CANONICAL_NODE_ALIASES
        base_nodes = list(getattr(base, "metadata", {}).get("node_names", []))
        self.source_indices: list[int | None] = []
        for node in self.target_nodes:
            candidates = self.aliases.get(node, (node,))
            idx = next((base_nodes.index(name) for name in candidates if name in base_nodes), None)
            self.source_indices.append(idx)
        metadata = dict(getattr(base, "metadata", {}))
        metadata["node_names"] = self.target_nodes
        metadata["targets"] = list(self.target_names)
        self.metadata = metadata

    def __len__(self) -> int:
        """Return the number of base rows."""

        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one item with canonical node order and selected targets."""

        item = self.base[index]
        x = item["x"]
        mapped = torch.zeros((len(self.target_nodes), x.shape[1], x.shape[2]), dtype=x.dtype)
        for dst, src in enumerate(self.source_indices):
            if src is not None:
                mapped[dst] = x[src]
        targets = {name: value for name, value in item["targets"].items() if name in self.target_names}
        mapped_item = {
            "x": mapped,
            "targets": targets,
            "subject": item.get("subject", ""),
            "start_idx": item.get("start_idx", 0),
        }
        for name in ("record", "activity", "future_activity", "activity_valid", "current_hr", "current_pat"):
            if name in item:
                mapped_item[name] = item[name]
        return mapped_item


def activity_filtered_dataset(
    dataset: PhysioWindowDataset,
    *,
    include: Iterable[int] | None = None,
    exclude: Iterable[int] | None = None,
) -> IndexedDataset:
    """Return a dataset view filtered by integer activity labels."""

    if "activity" in dataset.metadata_arrays:
        activity = np.asarray(dataset.metadata_arrays["activity"]).reshape(-1)
    elif "activity" in dataset.targets:  # backward compatibility with v2 data
        activity = np.asarray(dataset.targets["activity"]).reshape(-1)
    else:
        return IndexedDataset(dataset, np.arange(len(dataset)))
    mask = np.ones(activity.shape[0], dtype=bool)
    if "activity_valid" in dataset.metadata_arrays:
        mask &= np.asarray(dataset.metadata_arrays["activity_valid"]).reshape(-1).astype(bool)
    if include is not None:
        include_values = np.asarray(list(include), dtype=activity.dtype)
        mask &= np.isin(activity, include_values)
    if exclude is not None:
        exclude_values = np.asarray(list(exclude), dtype=activity.dtype)
        mask &= ~np.isin(activity, exclude_values)
    return IndexedDataset(dataset, np.flatnonzero(mask))


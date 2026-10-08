"""ONNX export, quantization, and CPU latency benchmarking."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


class _OnnxOutputWrapper(torch.nn.Module):
    """Wrap dictionary-output models as tuple-output modules for ONNX."""

    def __init__(self, model: torch.nn.Module, output_keys: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.output_keys = output_keys

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return configured tensor outputs in a stable order."""

        out = self.model(x)
        return tuple(out[key] for key in self.output_keys if key in out and isinstance(out[key], torch.Tensor))


def export_onnx(
    model: torch.nn.Module,
    example: torch.Tensor,
    output_path: str | Path,
    output_keys: tuple[str, ...] = ("wave", "hr", "bp", "stress"),
) -> Path:
    """Export a model to ONNX with stable tensor outputs."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = _OnnxOutputWrapper(model, output_keys).eval()
    torch.onnx.export(
        wrapper,
        example,
        output_path,
        input_names=["x"],
        output_names=list(output_keys),
        opset_version=17,
        dynamic_axes={"x": {0: "batch", 2: "time"}},
    )
    return output_path


def quantize_onnx_dynamic(onnx_path: str | Path, output_path: str | Path) -> Path:
    """Apply ONNX Runtime dynamic INT8 quantization."""

    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    quantize_dynamic(str(onnx_path), str(output_path), weight_type=QuantType.QInt8)
    return output_path


def benchmark_torch_latency(model: torch.nn.Module, example: torch.Tensor, warmup: int = 10, runs: int = 100) -> dict[str, float]:
    """Benchmark PyTorch CPU latency in milliseconds."""

    model.eval().cpu()
    example = example.cpu()
    with torch.no_grad():
        for _ in range(warmup):
            model(example)
        start = time.perf_counter()
        for _ in range(runs):
            model(example)
        elapsed = time.perf_counter() - start
    return {"latency_ms": elapsed * 1000.0 / max(runs, 1)}


def write_deployment_report(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write deployment benchmark rows as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

"""On-device efficiency benchmark: params, CPU latency, FP32/INT8 size.

Real, model-output-independent measurements for the deployment table. Runs on
CPU only so it does not interfere with concurrent GPU training.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.train import build_model

ROOT = Path(__file__).resolve().parents[1]
META = json.load(open(ROOT / "data/processed/ppg_dalia/metadata.json"))
NODES = len(META["node_names"])
T = int(META["input_steps"])

MODELS = {
    "PI-CDGN": {"name": "picdgn", "d_model": 128, "chunk_sec": 2, "message_passing_steps": 2,
                 "tau_min_ms": 0, "tau_max_ms": 800, "heads": {"waveform": True, "hr": True, "bp": True, "stress": True},
                 "ablations": {}},
    "LSTM": {"name": "lstm", "hidden_size": 128},
    "PatchTST": {"name": "patchtst", "hidden_size": 128},
    "Graph WaveNet": {"name": "graph_wavenet", "hidden_size": 128, "dilations": [1, 2, 4, 8], "adaptive_adj": True},
    "CorNET": {"name": "cornet", "hidden_size": 128},
}


def size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def cpu_latency_ms(model: torch.nn.Module, x: torch.Tensor, warmup: int = 3, runs: int = 20) -> float:
    model.eval().cpu()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
    return (time.perf_counter() - start) * 1000.0 / runs


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, NODES, T, 1)
    rows = []
    for label, mc in MODELS.items():
        model = build_model(OmegaConf.create({"model": mc}), META)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        fp32 = size_mb(model)
        try:
            qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            int8 = size_mb(qmodel)
        except Exception:
            int8 = float("nan")
        lat = cpu_latency_ms(model, x)
        rows.append({"model": label, "params_m": round(params_m, 3), "fp32_mb": round(fp32, 2),
                     "int8_mb": round(int8, 2), "cpu_latency_ms": round(lat, 2)})
        print(f"{label:14s} params={params_m:6.3f}M  fp32={fp32:6.2f}MB  int8={int8:6.2f}MB  cpu={lat:7.2f}ms/window")
    out = ROOT / "paper/data/deploy_metrics.json"
    json.dump(rows, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()

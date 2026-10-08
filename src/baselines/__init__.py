"""Baseline model registry."""

from __future__ import annotations

from src.baselines.agcrn import AGCRN
from src.baselines.bpnet import BPNet
from src.baselines.cornet import CorNET
from src.baselines.crossgnn import CrossGNN
from src.baselines.deep_ppg import DeepPPG
from src.baselines.fouriergnn import FourierGNN
from src.baselines.gwn import GraphWaveNet
from src.baselines.itransformer import ITransformer
from src.baselines.lstm import LSTM
from src.baselines.mtgnn import MTGNN
from src.baselines.patchtst import PatchTST
from src.baselines.stgcn import STGCN
from src.baselines.timemixer import TimeMixer
from src.baselines.timesnet import TimesNet
from src.baselines.transformer import Transformer

BASELINE_REGISTRY = {
    "lstm": LSTM,
    "transformer": Transformer,
    "patchtst": PatchTST,
    "itransformer": ITransformer,
    "timemixer": TimeMixer,
    "timesnet": TimesNet,
    "stgcn": STGCN,
    "gwn": GraphWaveNet,
    "mtgnn": MTGNN,
    "agcrn": AGCRN,
    "crossgnn": CrossGNN,
    "fouriergnn": FourierGNN,
    "cornet": CorNET,
    "deep_ppg": DeepPPG,
    "bpnet": BPNet,
}

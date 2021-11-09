from dataclasses import dataclass

from .none import NoneLayer
from .lstm import LSTMLayer
from .middle_layer import MiddleLayer

@dataclass
class MiddleLayerFactories:
    NONE = NoneLayer
    LSTM = LSTMLayer

    get_layer = {
        "NONE": NONE,
        "LSTM": LSTM
    }

__all__ = [
    "MiddleLayerFactories",
    "MiddleLayer"
]
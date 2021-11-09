from dataclasses import dataclass

from .none import NoneLayer
from .fully_connected import FullyConnectedLayer
from .output_layer import OutputLayer

@dataclass
class OutputLayerFactories:
    NONE = NoneLayer
    FC = FullyConnectedLayer

    get_layer = {
        "NONE": NONE,
        "FC": FC
    }

__all__ = [
    "OutputLayerFactories",
    "OutputLayer"
]
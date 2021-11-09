import torch.nn as nn

class LSTMLayer(nn.Module):
    def __init__(self, parameters):
        super(LSTMLayer, self).__init__()

    def __str__(self):
        return "lstm"

    def forward(self, features, parameters):
        raise NotImplementedError()
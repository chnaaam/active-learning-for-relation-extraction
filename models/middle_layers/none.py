import torch.nn as nn

class NoneLayer(nn.Module):
    def __init__(self, parameters):
        super(NoneLayer, self).__init__()

    def __str__(self):
        return "none"

    def forward(self, features, parameters=None):
        return features